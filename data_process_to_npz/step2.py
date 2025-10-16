import os
import numpy as np
import neurokit2 as nk
from scipy.signal import savgol_filter
import biobss
import argparse
import logging
from typing import Optional

# -----------------------------
# Quality assessment and dataset wrapper (with minor robustness tweaks)
# -----------------------------
class ECGDataset:
    def __init__(self, file_path: str):
        data = np.load(file_path, allow_pickle=True)
        self.file_name = data["file_name"]
        self.ppg_data = data["PPG"]  # shape (N, L) or (N,)
        self.ecg_data = data["ECG"]

        # ensure 2D shape (N, L)
        if self.ppg_data.ndim == 1: self.ppg_data = self.ppg_data[None, :]
        if self.ecg_data.ndim == 1: self.ecg_data = self.ecg_data[None, :]

    def __len__(self):
        return len(self.ecg_data)

    # ---------- PPG quality control ----------
    def ppg_quality_checker(self, data: np.ndarray, sr: float, delta: float = 1e-4,
                            correct_peaks: bool = True, sim_thre: float = 0.8, frac_thre: float = 0.8):
        assert data.ndim in (1, 2)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        check_result = []
        for item in data:
            try:
                # 1) baseline drift (rough)
                baseline_drift = np.abs(np.mean(np.diff(item)))
                baseline_ok = baseline_drift < 0.1

                # 2) SNR
                smooth = _safe_savgol(item, window_length=9, polyorder=2)
                noise = item - smooth
                signal_power = np.mean(item ** 2)
                noise_power = np.mean(noise ** 2)
                snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100.0
                snr_ok = snr > 10

                # 3) peaks and intervals
                info = biobss.ppgtools.ppg_detectpeaks(sig=item, sampling_rate=sr,
                                                       method='peakdet', delta=delta, correct_peaks=correct_peaks)
                locs_peaks = info['Peak_locs']
                if len(locs_peaks) > 1:
                    intervals = np.diff(locs_peaks)
                    cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
                    interval_ok = (cv < 0.5) and (0.5 * sr < np.mean(intervals) < 2.0 * sr)
                else:
                    interval_ok = False

                # 4) template-matching SQI
                sim, _ = biobss.sqatools.template_matching(item, locs_peaks)
                sim = np.asarray(sim)
                ppg_sqi = (sim >= sim_thre).sum(axis=-1) / (sim.shape[-1] + 1e-8) if sim.ndim > 0 else 0.0
                template_ok = ppg_sqi > frac_thre

                # 5) amplitude
                amplitude_ok = 0.1 < (np.max(item) - np.min(item)) < 10.0

                checks = [baseline_ok, snr_ok, interval_ok, template_ok, amplitude_ok]
                sqi_flag = (np.mean(checks) >= 0.8)
            except Exception as e:
                sqi_flag = False
            check_result.append(sqi_flag)
        return np.asarray(check_result, dtype=bool)

    # ---------- ECG quality control ----------
    def ecg_quality_checker(self, data: np.ndarray, sr: float,
                            sqi_thre: float = 0.95, frac_thre: float = 0.8, method: str = 'custom'):
        assert data.ndim in (1, 2)
        assert method in ['custom', 'neurokit']
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        check_result = []
        if method == 'neurokit':
            for item in data:
                try:
                    item_sqi = nk.ecg_quality(item, sampling_rate=int(sr), method='zhao2018')
                    check_result.append(item_sqi != 'Unacceptable')
                except Exception:
                    check_result.append(False)
        else:
            from scipy.fft import fft

            for item in data:
                try:
                    scores = []

                    # 3) R-peak count
                    ecg_cleaned = nk.ecg_clean(item, sampling_rate=sr, method="neurokit")
                    rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sr)[1]['ECG_R_Peaks']
                    min_expected_peaks = int(len(item) / sr * 0.5)  # ~30 bpm
                    scores.append(1.0 if len(rpeaks) >= min_expected_peaks else 0.0)

                    # 4) RR intervals
                    if len(rpeaks) > 3:
                        rr = np.diff(rpeaks) / sr
                        rr_ok = (0.4 < np.mean(rr) < 1.5) and ((np.std(rr) / (np.mean(rr) + 1e-8)) < 0.2)
                        scores.append(1.0 if rr_ok else 0.0)
                    else:
                        scores.append(0.0)

                    # 5) amplitude
                    amp_ok = 0.2 < (np.max(item) - np.min(item)) < 5.0
                    scores.append(1.0 if amp_ok else 0.0)

                    # 6) spectral energy ratio (0.5–40 Hz)
                    freqs = np.fft.fftfreq(len(item), d=1/sr)
                    spec = np.abs(fft(item))
                    valid = (np.abs(freqs) >= 0.5) & (np.abs(freqs) <= 40)
                    ratio = (spec[valid].sum() / (spec.sum() + 1e-12)) if spec.sum() > 0 else 0.0
                    scores.append(1.0 if ratio > sqi_thre else 0.0)

                    # 7) QRS template consistency
                    if len(rpeaks) > 3:
                        segs = []
                        half = int(0.1 * sr)
                        for p in rpeaks[1:-1]:
                            if p - half >= 0 and p + half < len(item):
                                segs.append(item[p - half: p + half])
                        if len(segs) > 2:
                            segs = np.asarray(segs)
                            ref = segs.mean(axis=0)
                            cors = [np.corrcoef(ref, s)[0, 1] for s in segs]
                            scores.append(1.0 if (np.mean(cors) > sqi_thre) else 0.0)
                        else:
                            scores.append(0.0)
                    else:
                        scores.append(0.0)

                    check_result.append(np.mean(scores) >= frac_thre)
                except Exception:
                    check_result.append(False)

        return np.asarray(check_result, dtype=bool)


# -----------------------------
# Utility functions
# -----------------------------
def _safe_savgol(x: np.ndarray, window_length: int, polyorder: int):
    """Automatically adjust window length to be odd and not exceed signal length; return original if too short."""
    L = len(x)
    if L < polyorder + 2:
        return x
    w = min(window_length, L if L % 2 == 1 else L - 1)
    if w < polyorder + 2:
        w = polyorder + 3
        if w % 2 == 0:
            w += 1
        if w > L:
            return x
    return savgol_filter(x, window_length=w, polyorder=polyorder) 

def _batch_savgol(arr: np.ndarray, win: int, poly: int):
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        out[i] = _safe_savgol(arr[i], win, poly)
    return out


# -----------------------------
# Main pipeline: quality filtering + save
# -----------------------------
def process_and_save(input_npz: str,
                     output_npz: Optional[str] = None,
                     orig_sr: float = 128.0,
                     target_sr: float = 128.0,
                     ppg_win: int = 7,
                     ppg_poly: int = 2,
                     ecg_win: int = 11,
                     ecg_poly: int = 2,
                     ecg_method: str = 'custom'):
    logging.info(f"Loading: {input_npz}")
    ds = ECGDataset(input_npz)
    file_name = ds.file_name
    PPG = ds.ppg_data.astype(np.float32)  # (N, L)
    ECG = ds.ecg_data.astype(np.float32)  # (N, L)
    assert PPG.shape[0] == ECG.shape[0], "PPG 与 ECG 条目数不一致"

    # S-G smoothing
    logging.info("Applying Savitzky–Golay smoothing ...")
    PPG_s = _batch_savgol(PPG, ppg_win, ppg_poly)
    ECG_s = _batch_savgol(ECG, ecg_win, ecg_poly)

    # Sample-level quality control (returns (N,) bool)
    logging.info("Running PPG quality check ...")
    ppg_flags = ds.ppg_quality_checker(PPG_s, sr=target_sr)
    ecg_flags = ds.ecg_quality_checker(ECG_s, sr=target_sr, method=ecg_method)

    keep_mask = ppg_flags & ecg_flags
    kept = int(keep_mask.sum())
    total = len(keep_mask)
    logging.info(f"Kept {kept}/{total} samples ({kept/total:.1%}).")

    file_name = file_name[keep_mask]
    PPG_kept = PPG_s[keep_mask]
    ECG_kept = ECG_s[keep_mask]
    print(PPG_kept.shape)

    # Save
    if output_npz is None:
        base, ext = os.path.splitext(input_npz)
        output_npz = base + "_filtered.npz"
    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)

    np.savez_compressed(output_npz, file_name=file_name,PPG=PPG_kept, ECG=ECG_kept)
    logging.info(f"Saved to: {output_npz}")
    return output_npz, kept, total


def parse_args():
    p = argparse.ArgumentParser(description="PPG/ECG 质量筛选并保存为 npz")
    p.add_argument("--input", type=str, required=True, help="输入 npz 路径（含 PPG/ECG 两个键）")
    p.add_argument("--output", type=str, default=None, help="输出 npz 路径，默认自动加 _filtered 后缀")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_and_save(input_npz=args.input,
                     output_npz=args.output)
