import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity
from dtw import dtw as dtw_metric
import os
import sys

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

from utils.data import ECGDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from typing import Optional, Tuple
from biosppy.signals import ecg as ecg_func
from biosppy.signals import ppg as ppg_func
from biosppy.signals import tools
import neurokit2 as nk

def calculate_mae(gt, fake):
    """Mean Absolute Error (MAE)."""
    return np.mean(np.abs(gt - fake))

def calculate_rmse(gt, fake):
    """Root Mean Square Error (RMSE)."""
    return np.sqrt(np.mean((gt - fake) ** 2))

def calculate_fd_for_small_sample(
    gt: np.ndarray, 
    fake: np.ndarray, 
    pca_dim: Optional[int] = 64,
    eps: float = 1e-4,
    n_trials: int = 1
) -> Tuple[float, float]:
    """
    FD computation optimized for small-sample regimes.
    Args:
        gt: ground-truth data
        fake: generated data
        pca_dim: target PCA dimension; None means no PCA
        eps: numerical stability term (suggest larger like 1e-4 for small samples)
        n_trials: number of trials; when > 1 returns mean and std
    Returns:
        mean_fd: mean FD across trials
        std_fd: std of FD (0 when n_trials == 1)
    """
    fds = []
    for _ in range(n_trials):
        # Flatten each sample
        gt_flat = gt.reshape(gt.shape[0], -1)
        fake_flat = fake.reshape(fake.shape[0], -1)
        
        # Optional PCA reduction
        if pca_dim is not None:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=pca_dim, svd_solver="auto", whiten=False)
            data_combined = np.concatenate([gt_flat, fake_flat], axis=0)
            data_reduced = pca.fit_transform(data_combined)
            gt_flat = data_reduced[:gt.shape[0]]
            fake_flat = data_reduced[gt.shape[0]:]
        
        # Compute means and covariances
        mu_gt = gt_flat.mean(axis=0)
        mu_fake = fake_flat.mean(axis=0)
        cov_gt = np.cov(gt_flat, rowvar=False)
        cov_fake = np.cov(fake_flat, rowvar=False)
        
        # Numerical stabilization
        dim = cov_gt.shape[0]
        cov_gt = cov_gt + eps * np.eye(dim)
        cov_fake = cov_fake + eps * np.eye(dim)
        
        # Compute FD
        ssdiff = np.sum((mu_gt - mu_fake) ** 2.0)
        covmean, _ = sqrtm(cov_gt.dot(cov_fake), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fd = float(ssdiff + np.trace(cov_gt) + np.trace(cov_fake) - 2.0 * np.trace(covmean))
        fds.append(fd)
    
    return np.mean(fds), np.std(fds)

def calculate_fd(gt: np.ndarray, fake: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Fréchet Distance (FD) in raw signal space (flatten each sample).
    Adds numerical stability term eps and uses disp=False to suppress sqrtm output.
    """
    gt = np.asarray(gt, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)

    gt_flat = gt.reshape(gt.shape[0], -1)
    fake_flat = fake.reshape(fake.shape[0], -1)

    mu_gt = gt_flat.mean(axis=0)
    mu_fake = fake_flat.mean(axis=0)

    cov_gt = np.cov(gt_flat, rowvar=False)
    cov_fake = np.cov(fake_flat, rowvar=False)

    # Numerical stabilization
    dim = cov_gt.shape[0]
    cov_gt = cov_gt + eps * np.eye(dim)
    cov_fake = cov_fake + eps * np.eye(dim)

    ssdiff = np.sum((mu_gt - mu_fake) ** 2.0)
    covmean, _ = sqrtm(cov_gt.dot(cov_fake), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd_value = float(ssdiff + np.trace(cov_gt) + np.trace(cov_fake) - 2.0 * np.trace(covmean))
    return fd_value

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class BasicBlock(nn.Module):
    """
    Basic Block:
        conv1 -> convk -> conv1

    params:
        in_channels: number of input channels
        out_channels: number of output channels
        ratio: ratio of channels to out_channels
        kernel_size: kernel window length
        stride: kernel step size
        groups: number of groups in convk
        downsample: whether downsample length
        use_bn: whether use batch_norm
        use_do: whether use dropout

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        ratio,
        kernel_size,
        stride,
        groups,
        downsample,
        is_first_block=False,
        use_bn=True,
        use_do=True,
    ):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.middle_channels = int(self.out_channels * self.ratio)

        # the first conv, conv1
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=self.in_channels,
            out_channels=self.middle_channels,
            kernel_size=1,
            stride=1,
            groups=1,
        )

        # the second conv, convk
        self.bn2 = nn.BatchNorm1d(self.middle_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=self.middle_channels,
            out_channels=self.middle_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

        # the third conv, conv1
        self.bn3 = nn.BatchNorm1d(self.middle_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(
            in_channels=self.middle_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
        )

        # Squeeze-and-Excitation
        r = 2
        self.se_fc1 = nn.Linear(self.out_channels, self.out_channels // r)
        self.se_fc2 = nn.Linear(self.out_channels // r, self.out_channels)
        self.se_activation = Swish()

        if self.downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):

        identity = x

        out = x
        # the first conv, conv1
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv, convk
        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # the third conv, conv1
        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out)  # (n_sample, n_channel, n_length)

        # Squeeze-and-Excitation
        se = out.mean(-1)  # (n_sample, n_channel)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = torch.sigmoid(se)  # (n_sample, n_channel)
        out = torch.einsum("abc,ab->abc", out, se)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out

class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding

    params:
        kernel_size: kernel size
        stride: the stride of the window. Default value is kernel_size

    input: (n_sample, n_channel, n_length)
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):

        net = x

        # compute pad shape
        p = max(0, self.kernel_size - 1)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


# Remove redundant definitions comment

class BasicStage(nn.Module):
    """
    Basic Stage:
        block_1 -> block_2 -> ... -> block_M
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        ratio,
        kernel_size,
        stride,
        groups,
        i_stage,
        m_blocks,
        use_bn=True,
        use_do=True,
        verbose=False,
    ):
        super(BasicStage, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.i_stage = i_stage
        self.m_blocks = m_blocks
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        self.block_list = nn.ModuleList()
        for i_block in range(self.m_blocks):

            # first block
            if self.i_stage == 0 and i_block == 0:
                self.is_first_block = True
            else:
                self.is_first_block = False
            # downsample, stride, input
            if i_block == 0:
                self.downsample = True
                self.stride = stride
                self.tmp_in_channels = self.in_channels
            else:
                self.downsample = False
                self.stride = 1
                self.tmp_in_channels = self.out_channels

            # build block
            tmp_block = BasicBlock(
                in_channels=self.tmp_in_channels,
                out_channels=self.out_channels,
                ratio=self.ratio,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=self.downsample,
                is_first_block=self.is_first_block,
                use_bn=self.use_bn,
                use_do=self.use_do,
            )
            self.block_list.append(tmp_block)

    def forward(self, x):

        out = x

        for i_block in range(self.m_blocks):
            net = self.block_list[i_block]
            out = net(out)

        return out

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net

class Net1D(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    params:
        in_channels
        base_filters
        filter_list: list, filters for each stage
        m_blocks_list: list, number of blocks of each stage
        kernel_size
        stride
        groups_width
        n_stages
        n_classes
        use_bn
        use_do

    """

    def __init__(
        self,
        in_channels,
        base_filters,
        ratio,
        filter_list,
        m_blocks_list,
        kernel_size,
        stride,
        groups_width,
        n_classes,
        use_bn=True,
        use_do=True,
        verbose=False,
    ):
        super(Net1D, self).__init__()

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.ratio = ratio
        self.filter_list = filter_list
        self.m_blocks_list = m_blocks_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups_width = groups_width
        self.n_stages = len(filter_list)
        self.n_classes = n_classes
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        # first conv
        self.first_conv = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=self.base_filters,
            kernel_size=self.kernel_size,
            stride=2,
        )
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()

        # stages
        self.stage_list = nn.ModuleList()
        in_channels = self.base_filters
        for i_stage in range(self.n_stages):

            out_channels = self.filter_list[i_stage]
            m_blocks = self.m_blocks_list[i_stage]
            tmp_stage = BasicStage(
                in_channels=in_channels,
                out_channels=out_channels,
                ratio=self.ratio,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=out_channels // self.groups_width,
                i_stage=i_stage,
                m_blocks=m_blocks,
                use_bn=self.use_bn,
                use_do=self.use_do,
                verbose=self.verbose,
            )
            self.stage_list.append(tmp_stage)
            in_channels = out_channels

        # final prediction
        self.dense = nn.Linear(in_channels, n_classes)

    def forward(self, x):

        out = x

        # first conv
        out = self.first_conv(out)
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)

        # stages
        for i_stage in range(self.n_stages):
            net = self.stage_list[i_stage]
            out = net(out)

        # final prediction
        out = out.mean(-1)
        out = self.dense(out)

        return out

def ft_1lead_ECGFounder(
    device: torch.device,
    n_classes: int = 1000,
    use_bn: bool = True,
    use_do: bool = False,
) -> nn.Module:
    model = Net1D(
        in_channels=1, 
        base_filters=64, #32 64
        ratio=1, 
        filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
        m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
        kernel_size=16, 
        stride=2, 
        groups_width=16,
        verbose=False, 
        use_bn=use_bn,
        use_do=use_do,
        n_classes=n_classes)

    checkpoint = torch.load("1_lead_ECGFounder.pth", map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']

    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

    model.load_state_dict(state_dict, strict=False)

    model.dense = nn.Identity()

    model.to(device)

    return model

def compute_representations_in_batches(
    model: nn.Module,
    data: np.ndarray,
    batch_size: int = 128,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_workers: int = 4,
) -> torch.Tensor:
    dataset = TensorDataset(torch.from_numpy(data).float().transpose(1, 2))
    pin_memory = isinstance(device, torch.device) and device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    representations = []
    with torch.inference_mode():
        for (batch_data,) in dataloader:
            batch_rep = model(batch_data.to(device, non_blocking=True))
            representations.append(batch_rep.cpu())

    return torch.cat(representations, dim=0)

def _maybe_apply_pca(M1: np.ndarray, M2: np.ndarray, pca_dim: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if pca_dim is None:
        return M1, M2
    feature_dim = M1.shape[1]
    if pca_dim <= 0 or pca_dim >= feature_dim:
        return M1, M2
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        print(f"[WARN] sklearn not installed, skip PCA. Error: {e}")
        return M1, M2
    # Fit PCA on concatenated features to keep both sides consistent
    pca = PCA(n_components=pca_dim, svd_solver="auto", whiten=False)
    M_all = np.concatenate([M1, M2], axis=0)
    M_all_reduced = pca.fit_transform(M_all)
    M1_reduced = M_all_reduced[: M1.shape[0]]
    M2_reduced = M_all_reduced[M1.shape[0] :]
    return M1_reduced, M2_reduced


def calculate_FID_score(
    M1: torch.Tensor,
    M2: torch.Tensor,
    eps: float = 1e-6,
    pca_dim: Optional[int] = None,
) -> float:
    """
    Compute a Fréchet Inception Distance-style score in representation space.
    - Adds eps for covariance stability
    - Optional PCA on concatenated real/fake features
    """
    M1_np = M1.cpu().numpy().astype(np.float64)
    M2_np = M2.cpu().numpy().astype(np.float64)

    # Optional dimensionality reduction
    M1_np, M2_np = _maybe_apply_pca(M1_np, M2_np, pca_dim)

    mu1, sigma1 = M1_np.mean(axis=0), np.cov(M1_np, rowvar=False)
    mu2, sigma2 = M2_np.mean(axis=0), np.cov(M2_np, rowvar=False)

    # Numerical stabilization
    dim = sigma1.shape[0]
    sigma1 = sigma1 + eps * np.eye(dim)
    sigma2 = sigma2 + eps * np.eye(dim)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


def zscore_per_sample(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-sample z-score along time for (N, L) or (N, L, C) arrays.
    """
    if x.ndim == 2:
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + eps
        return (x - mean) / std
    elif x.ndim == 3:
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + eps
        return (x - mean) / std
    else:
        raise ValueError(f"Unsupported input dimensions: {x.ndim}")

# -------------------- Heart-rate related evaluation (ECG/PPG) --------------------

def get_Rpeaks_ECG(filtered, sampling_rate):
    # segment
    rpeaks, = ecg_func.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    rpeaks, = ecg_func.correct_rpeaks(
        signal=filtered,
        rpeaks=rpeaks,
        sampling_rate=sampling_rate,
        tol=0.05,
    )

    # extract templates
    templates, rpeaks = ecg_func.extract_heartbeats(
        signal=filtered,
        rpeaks=rpeaks,
        sampling_rate=sampling_rate,
        before=0.2,
        after=0.4,
    )

    rr_intervals = np.diff(rpeaks)

    return rpeaks, rr_intervals


def get_peaks_PPG(filtered, sampling_rate=128):
    # segment
    peaks = ppg_func.ppg_findpeaks(filtered, sampling_rate)["PPG_Peaks"]
    peak_intervals = np.diff(peaks)

    return peaks, peak_intervals


def heartbeats_ecg(filtered, sampling_rate):
    rpeaks, rr_intervals = get_Rpeaks_ECG(filtered, sampling_rate)

    if rr_intervals.size != 0:
        # compute heart rate
        hr_idx, hr = tools.get_heart_rate(
            beats=rpeaks, sampling_rate=sampling_rate, smooth=True, size=3
        )

        if len(hr) == 0:
            hr_idx, hr = [-1], [-1]

    else:
        hr_idx, hr = [-1], [-1]

    return hr_idx, hr


def heartbeats_ppg(filtered, sampling_rate):
    peaks, peaks_intervals = get_peaks_PPG(filtered, sampling_rate)

    if peaks_intervals.size != 0:
        # compute heart rate
        hr_idx, hr = tools.get_heart_rate(
            beats=peaks, sampling_rate=sampling_rate, smooth=True, size=3
        )

        if len(hr) == 0:
            hr_idx, hr = [-1], [-1]

    else:
        hr_idx, hr = [-1], [-1]

    return hr_idx, hr


def ecg_bpm_array(ecg_signal, sampling_rate=128, window=4, filter=False):
    final_bpm = []
    for k in ecg_signal:
        # support (L,) or (L,1)
        if isinstance(k, np.ndarray) and k.ndim > 1:
            k = np.squeeze(k)
        if filter is True:
            k = nk.ecg_clean(k, sampling_rate=128, method="pantompkins1985")
        hr_idx, hr = heartbeats_ecg(k, sampling_rate)
        bpm = np.mean(hr)
        final_bpm.append(bpm)
    return np.array(final_bpm)


def ppg_bpm_array(ppg_signal, sampling_rate=128, window=4):
    final_bpm = []
    for k in ppg_signal:
        try:
            # support (L,) or (L,1)
            if isinstance(k, np.ndarray) and k.ndim > 1:
                k = np.squeeze(k)
            hr_idx, hr = heartbeats_ppg(k, sampling_rate)
            bpm = np.mean(hr)
            final_bpm.append(bpm)
        except Exception:
            final_bpm.append(-1.0)

    return np.array(final_bpm)


def MAE_hr(real_ecg, fake_ecg, ecg_sampling_freq=128, window_size=4):
    # HR estimation from Fake ECG and Real ECG
    real_ecg_bpm = ecg_bpm_array(real_ecg, ecg_sampling_freq, window_size)
    fake_ecg_bpm = ecg_bpm_array(
        fake_ecg, ecg_sampling_freq, window_size, filter=True
    )  # check for -1 values

    # correction
    valid = (~np.isnan(fake_ecg_bpm)) & (~np.isnan(real_ecg_bpm))
    fbpm = fake_ecg_bpm[valid]
    rbpm = real_ecg_bpm[valid]
    # fbpm = fake_ecg_bpm[np.where(fake_ecg_bpm != -1)]
    # rbpm = real_ecg_bpm[np.where(fake_ecg_bpm != -1)]

    mae_hr_ecg = np.mean(np.absolute(rbpm - fbpm))

    return mae_hr_ecg

# Example usage
if __name__ == "__main__":
    # Load ground-truth and generated data
    gt = np.load("")
    fake = np.load("")
    # Ensure shape (N, L, 1)
    if gt.ndim == 2:
        gt = np.expand_dims(gt, axis=-1)
    if fake.ndim == 2:
        fake = np.expand_dims(fake, axis=-1)

    # Optional normalization and PCA settings
    ENABLE_NORMALIZE = False
    
    if ENABLE_NORMALIZE:
        gt = zscore_per_sample(gt)
        fake = zscore_per_sample(fake)
    print(f"GT data shape: {gt.shape}")
    print(f"Fake data shape: {fake.shape}")

    # Compute metrics
    mae = calculate_mae(gt, fake)
    print(f"MAE: {mae:.2f}")
    
    rmse = calculate_rmse(gt, fake)
    print(f"RMSE: {rmse:.2f}")

    # Choose FD strategy based on sample size
    n_samples = len(gt)
    if n_samples < 3000:
        # Small-sample regime
        pca_dim = 32
        eps = 1e-4
        n_trials = 5
        
        fd_mean, fd_std = calculate_fd_for_small_sample(
            gt, fake, 
            pca_dim=pca_dim,
            eps=eps,
            n_trials=n_trials
        )
        print(f"Frechet Distance (FD) for small sample: {fd_mean:.2f}")
    else:
        # Large-sample regime uses the original method
        fd = calculate_fd(gt, fake)
        print(f"Frechet Distance (FD): {fd:.2f}")

    # representation model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ft_1lead_ECGFounder(device, use_bn=True, use_do=False)
    model.eval()
    
    # compute FID
    fake_ecg_representation = compute_representations_in_batches(
        model, fake, batch_size=128, device=device
    )
    real_ecg_representation = compute_representations_in_batches(
        model, gt, batch_size=128, device=device
    )
    print(f"Representations: fake={fake_ecg_representation.shape}, real={real_ecg_representation.shape}")

    # Choose FID PCA parameters by sample size if needed
    PCA_DIM = None
    fid = calculate_FID_score(
        fake_ecg_representation, 
        real_ecg_representation, 
        eps=1e-6,
        pca_dim = PCA_DIM
    )
    print(f"FID: {fid:.2f}")

    mae_hr = MAE_hr(gt, fake, ecg_sampling_freq=128, window_size=10)
    print(f"mae_hr: {mae_hr:.2f}")