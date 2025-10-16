import mne
import biobss
import numpy as np
import neurokit2 as nk
from pathlib import Path
from scipy.signal import resample_poly
from typing import Optional, Tuple, Dict, Any, List

class MCMEDProcessor:
    def __init__(
            self,
            dataset_dir: str, 
            seg_duration: float, 
            overlap_ratio: float, 
            save_dir: str,
            window_duration: float = 0.1, 
            change_thershold: float = 0.001,
            tolerate_threshold: float = 0.25,
            ppg_high_cutoff: float = 8,
            ppg_low_cutoff: float = 0.5,
            ecg_low_cutoff: float = 0.5,
            ecg_high_cutoff: Optional[float] = None,
            peak_loc_delta: float = 1,
            ecg_invert_thre: float = 0.5, 
            ecg_sqi_thre: float = 0.5, 
            ecg_frac_thre: float = 0.75,
            ecg_sqi_method: str = 'zhao2018', 
            ecg_sqi_approach: str = 'simple',
            ppg_correct_peaks: bool = False,
            ppg_sim_thre: float = 0.5, 
            ppg_frac_thre: float = 0.5,
            ppg_quality_check: bool = False,
            ppg_resample_sr: int = 128,
            ecg_resample_sr: int = 128,
            log_dir: str = './logs',
            num_workers: int = 32
            ):
        self.dataset_dir = Path(dataset_dir)
        self.seg_duration = seg_duration
        self.overlap_ratio = overlap_ratio
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.window_duration = window_duration
        self.change_thershold = change_thershold
        self.tolerate_threshold = tolerate_threshold
        self.ppg_high_cutoff = ppg_high_cutoff
        self.ppg_low_cutoff = ppg_low_cutoff
        self.ecg_low_cutoff = ecg_low_cutoff
        self.ecg_high_cutoff = ecg_high_cutoff
        self.peak_loc_delta = peak_loc_delta
        self.ecg_invert_thre = ecg_invert_thre
        self.ecg_sqi_thre  = ecg_sqi_thre
        self.ecg_frac_thre = ecg_frac_thre
        self.ecg_sqi_method = ecg_sqi_method
        self.ecg_sqi_approach = ecg_sqi_approach
        self.ppg_correct_peaks = ppg_correct_peaks
        self.ppg_sim_thre = ppg_sim_thre 
        self.ppg_frac_thre = ppg_frac_thre
        self.ppg_quality_check = ppg_quality_check
        self.ppg_resample_sr = ppg_resample_sr
        self.ecg_resample_sr = ecg_resample_sr
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers

    def process_individual_data(self, npz_path: str) -> Tuple:
        # loading data
        file_name, ppg_data, ecg_data = self.data_reader(npz_path)
        ecg_sr = 125
        ppg_sr = 25

        file_name = np.array([(item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)) for item in file_name], dtype=np.str_)
        ppg_data = np.array([item.astype(np.float32) for item in ppg_data])
        ecg_data = np.array([item.astype(np.float32) for item in ecg_data], dtype=object)
        ecg_data, invert_ratios = self.ecg_invert_checker(ecg_data, ecg_sr, self.ecg_invert_thre) # type: ignore

        # NaN and Inf filtering
        valid_mask = np.isfinite(ppg_data).all(axis=-1) & np.isfinite(ecg_data).all(axis=-1)
        ppg_data = ppg_data[valid_mask]
        ecg_data = ecg_data[valid_mask]
        file_name = file_name[valid_mask]
        print("NaN",ppg_data.shape)

        # flat detection
        ppg_flat_flag = self.data_flat_detector(ppg_data, ppg_sr, self.window_duration, self.change_thershold, self.tolerate_threshold)
        ecg_flat_flag = self.data_flat_detector(ecg_data, ecg_sr, self.window_duration, self.change_thershold, self.tolerate_threshold)
        valid_ppg_flat_indices = np.where(ppg_flat_flag == True)[0]
        valid_ecg_flat_indices = np.where(ecg_flat_flag == True)[0]
        valid_flat_indices = np.intersect1d(valid_ppg_flat_indices, valid_ecg_flat_indices)

        file_name, ppg_data, ecg_data = file_name[valid_flat_indices], ppg_data[valid_flat_indices], ecg_data[valid_flat_indices]
        print("flat",ppg_data.shape)

        # filtering
        ppg_data = self.ppg_clean_elgendi_mne(ppg_data, ppg_sr, self.ppg_low_cutoff, self.ppg_high_cutoff)
        ecg_data = self.ecg_clean_nk_mne(ecg_data, ecg_sr, self.ecg_low_cutoff, self.ecg_high_cutoff)
        print("filtering",ppg_data.shape)

        # PPG signal quality assessment
        if self.ppg_quality_check:
            ppg_quality_check = self.ppg_quality_checker(ppg_data, ppg_sr, self.peak_loc_delta, self.ppg_correct_peaks, self.ppg_sim_thre, self.ppg_frac_thre)
        else:
            ppg_quality_check = np.asarray([True] * ppg_data.shape[0])

        # ECG signal quality assessment
        ecg_quality_check = self.ecg_quality_checker(ecg_data, ecg_sr, self.ecg_sqi_thre, self.ecg_frac_thre, self.ecg_sqi_method, self.ecg_sqi_approach)
        valid_ppg_quality_indices = np.where(ppg_quality_check == True)[0]
        valid_ecg_quality_indices = np.where(ecg_quality_check == True)[0]
        valid_quality_indices = np.intersect1d(valid_ppg_quality_indices, valid_ecg_quality_indices)

        file_name, ppg_data, ecg_data = file_name[valid_quality_indices], ppg_data[valid_quality_indices], ecg_data[valid_quality_indices]
        print("quality assessment",ppg_data.shape)

        # resampling
        ppg_data = self.data_resampler(ppg_data, ppg_sr, self.ppg_resample_sr)
        ecg_data = self.data_resampler(ecg_data, ecg_sr, self.ecg_resample_sr)

        # normalization
        ppg_data, ecg_data = self.data_normalizer(ppg_data), self.data_normalizer(ecg_data)

        np.savez(self.save_dir.joinpath(f'{Path(npz_path).stem}.npz'), file_name=file_name, PPG=ppg_data, ECG=ecg_data)
        print(f'{npz_path} has been successfully saved.')

        return ppg_data, ecg_data, npz_path, len(ppg_data), invert_ratios, valid_ppg_flat_indices, valid_ecg_flat_indices, valid_ppg_quality_indices, valid_ecg_quality_indices

    @staticmethod
    def data_reader(npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        file_name = data['file_name']
        ppg_data = data['ppg']
        ecg_data = data['ecg']
        return file_name, ppg_data, ecg_data

    @staticmethod
    def ppg_clean_elgendi_mne(
        ppg_signal: np.ndarray, 
        sr: float = 125, 
        l_freq: float = 0.5, 
        h_freq: float = 8.0, 
        method: str = 'iir',
        iir_params: Dict[str, Any] = dict(order=3, ftype='butter', output='ba'),
        *args, **kwargs
        ):
        assert ppg_signal.ndim == 2, f'Expected data dimention is 2, got {ppg_signal.ndim}'
        if ppg_signal.ndim == 1:
            ppg_signal = ppg_signal[np.newaxis, :]

        filtered = mne.filter.filter_data(
            ppg_signal.astype(float), 
            sfreq=sr,
            l_freq=l_freq, 
            h_freq=h_freq,
            method=method,
            iir_params=iir_params,
            verbose=False,
            *args, **kwargs
        )
            
        return filtered.astype(np.float32)

    @staticmethod
    def ecg_clean_nk_mne(
        ecg_signal: np.ndarray, 
        sr: float = 500,
        l_freq: float = 0.5,
        h_freq: Optional[float] = None,
        method: str = 'iir',
        iir_params: Dict = dict(order=5, ftype='butter', output='ba'),
        notch_widths: float = 1,
        *args, **kwargs
        ):
        filtered = mne.filter.filter_data(
            ecg_signal.astype(float),
            sfreq=sr,
            l_freq=l_freq,
            h_freq=h_freq,
            method=method,
            iir_params=iir_params,
            verbose=False,
            *args, **kwargs,
        )
        
        # Notch filter for powerline (50Hz)
        filtered = mne.filter.notch_filter(
            filtered,
            Fs=sr,
            freqs=50,
            notch_widths=notch_widths,
            method=method,
            verbose=False
        )
        return filtered.astype(np.float32)

    def data_resampler(self, data: np.ndarray, sr: float, resample_sr: float):
        assert data.ndim == 1 or data.ndim == 2, f'Expected data dimention is 1 or 2, got {data.ndim}'
        resampled_data = resample_poly(data, resample_sr, sr, axis=-1)
        return resampled_data

    def ecg_invert_checker(self, ecg_signal: List[np.ndarray], sampling_rate: float = 500, threshold: float = 0.5):
        correct_ecg_signal, invert_ratios = [], []
        for item in ecg_signal:
            correct_item, invert_ratio, _ = self.record_ecg_invert_checker(item, sampling_rate, threshold)
            correct_ecg_signal.append(correct_item.astype(np.float32))
            invert_ratios.append(invert_ratio)
        correct_ecg_signal = np.vstack(correct_ecg_signal)
        invert_ratios = np.array(invert_ratios)
        return correct_ecg_signal, invert_ratios

    def record_ecg_invert_checker(self, ecg_signal: np.ndarray, sampling_rate: float = 500, threshold: float = 0.5):
        """ECG signal inversion

        Checks whether an ECG signal is inverted, and if so, corrects for this inversion.
        To automatically detect the inversion, the ECG signal is cleaned, the mean is subtracted,
        and with a rolling window of 2 seconds, the original value corresponding to the maximum
        of the squared signal is taken. If the median of these values is negative, it is
        assumed that the signal is inverted.
        """
        assert ecg_signal.ndim == 1 or ecg_signal.ndim == 2, f'Expected data dimention is 1 or 2, got {ecg_signal.ndim}'
        if ecg_signal.ndim == 1:
            ecg_signal = np.expand_dims(ecg_signal, axis=0)
        
        def _ecg_inverted(ecg_signal, sampling_rate=1000, window_time=2.0):
            """Checks whether an ECG signal is inverted."""
            ecg_cleaned = self.ecg_clean_nk_mne(ecg_signal, sampling_rate)

            # take the median of the original value of the maximum of the squared signal
            # over a window where we would expect at least one heartbeat
            med_max_squared = np.nanmedian(
                _roll_orig_max_squared(ecg_cleaned, window=int(window_time * sampling_rate))
            )
            # if median is negative, assume inverted
            return med_max_squared < 0

        def _roll_orig_max_squared(x, window=2000):
            """With a rolling window, takes the original value corresponding to the maximum of the squared signal."""
            x_rolled = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)
            # https://stackoverflow.com/questions/61703879/in-numpy-how-to-select-elements-based-on-the-maximum-of-their-absolute-values
            shape = np.array(x_rolled.shape)
            shape[-1] = -1
            return np.take_along_axis(x_rolled, np.square(x_rolled).argmax(-1).reshape(shape), axis=-1)

        was_inverted = []
        for item in ecg_signal:
            if _ecg_inverted(item, sampling_rate=int(sampling_rate)):
                was_inverted.append(True)
            else:
                was_inverted.append(False)
        invert_ratio = sum(was_inverted) / len(was_inverted)
        judge = invert_ratio > threshold
        if judge:
            return ecg_signal * -1, invert_ratio, judge
        return ecg_signal, invert_ratio, judge

    def ecg_quality_checker(self, data: np.ndarray, sr: float, sqi_thre: float = 0.5, frac_thre: float = 0.75, method: str = 'zhao2018', approach: str = 'simple'):
        assert data.ndim == 1 or data.ndim == 2, f'Expected data dimention is 1 or 2, got {data.ndim}'
        assert method in ['averageQRS', 'zhao2018'], f'Expected signal quality assessment method is averageQRS or zhao2018, got {method}'
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        ecg_sqi = []
        for item in data:
            try:
                item_sqi = nk.ecg_quality(item, sampling_rate=int(sr), method=method, approach=approach)
            except Exception as e:
                print(f'ECG_QUALITY_CHECKER Error: {e}.')
                item_sqi = np.zeros(data.shape[-1]) if method == 'averageQRS' else 'Unacceptable'
            ecg_sqi.append(item_sqi)
        ecg_sqi = np.asarray(ecg_sqi)
        check_result = []
        if method == 'averageQRS':
            ecg_sqi = (ecg_sqi >= sqi_thre).sum(axis=-1) / ecg_sqi.shape[-1]
            check_result = ecg_sqi >= frac_thre
        if method == 'zhao2018':
            for item_sqi in ecg_sqi:
                if item_sqi == 'Unacceptable':
                    check_result.append(False)
                else:
                    check_result.append(True)
        check_result = np.asarray(check_result)
        return check_result
    
    def ppg_quality_checker(self, data: np.ndarray, sr: float, delta: float = 1e-4, correct_peaks: bool = True, sim_thre: float = 0.5, frac_thre: float = 0.5):
        assert data.ndim == 1 or data.ndim == 2, f'Expected data dimention is 1 or 2, got {data.ndim}'
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        check_result = []
        for item in data:
            try:
                info = biobss.ppgtools.ppg_detectpeaks(sig=item, sampling_rate=sr, method='peakdet', delta=delta, correct_peaks=correct_peaks)
                locs_peaks = info['Peak_locs']
                sim, _ = biobss.sqatools.template_matching(item, locs_peaks) # Template matching 
                sim = np.asarray(sim)
                ppg_sqi = (sim >= sim_thre).sum(axis=-1) / sim.shape[-1]
                sqi_flag = ppg_sqi > frac_thre
            except Exception as e:
                print(f'PPG_QUALITY_CHECKER ERROR: {e}.')
                sqi_flag = False
            check_result.append(sqi_flag)
        check_result = np.asarray(check_result)
        return check_result

    def data_flat_detector(self, data: np.ndarray, sr: float, window_duration: float = 0.1, change_threshold: float = 0.005, tolerate_threshold: float = 0.25):
        assert data.ndim == 1 or data.ndim == 2, f'Expected data dimention is 1 or 2, got {data.ndim}'
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        window_size = int(window_duration * sr)
        window_flat_ratios = self.detect_flat_in_windows(data, window_size, change_threshold)
        detect_result = window_flat_ratios <= tolerate_threshold
        return detect_result

    def data_normalizer(self, data: np.ndarray, eps: float = 1e-8):
        mean, std = data.mean(axis=-1, keepdims=True), data.std(axis=-1, keepdims=True)
        normalized_data = (data - mean) / (std + eps)
        return normalized_data

    @staticmethod
    def detect_flat_in_windows(windows: np.ndarray, window_size: int = 50, threshold: float = 0.005) -> np.ndarray:
        """
        Detect flat regions in sliding windows of the input data.

        Args:
            windows (np.ndarray): 2D array where each row is a window of the input data.
            window_size (int): Size of the sliding window.
            threshold (float): Threshold for determining flat regions.

        Returns:
            np.ndarray: An array of flat ratios for each window.
        """
        # Compute the absolute difference between consecutive points in each window
        diff = np.abs(np.diff(windows, axis=1))

        # Compute the change rate using a sliding window convolution
        change_rate = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(window_size - 1), mode='valid') / (window_size - 1),
            axis=1,
            arr=diff
        )
        
        # Determine if each window is flat based on the threshold
        flat_windows = change_rate < threshold
        
        # Compute the ratio of flat regions in each window
        flat_ratios = np.mean(flat_windows, axis=1)
        return flat_ratios

if __name__ == "__main__":
    # Example config
    config = {
        "dataset_dir": '',
        "seg_duration": 10,
        "overlap_ratio": 0,
        "save_dir": '',
        "num_workers": 16,
        "pattern": '*.npz'
    }
    
    # Create processor instance
    processor = MCMEDProcessor(**{k: config[k] for k in config.keys() if k != 'pattern'})
    
    # Process sample data
    processor.process_individual_data("train.npz")
    processor.process_individual_data("val_data.npz")
    processor.process_individual_data("test.npz")