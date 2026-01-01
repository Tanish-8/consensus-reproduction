import os
import math
import json
import warnings
import numpy as np
import neurokit2 as nk

from functools import partial
from tqdm import tqdm
from fire import Fire
from glob import glob
from mne.io import read_raw_edf  # pylint: disable=no-name-in-module
from scipy import signal
from datasets import Dataset
from datetime import datetime
from multiprocessing import Pool
from dhedfreader import BaseEDFReader
from neurokit2.misc._warnings import NeuroKitWarning

warnings.simplefilter("ignore", NeuroKitWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

SLEEPEDF_PATH = "./sleepedf"  # path to the sleepedf dataset
OUT_DIR = "./sleepedf_output"  # path to the output directory
EPOCH_SEC_SIZE = 30
SAMPLING_RATE = 100

W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

class_dict = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 5: "UNKNOWN"}
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5,
}


def store_info(info_path):
    info = {
        "task": 'Classify the user\'s sleep stage: ["W", "N1", "N2", "N3", "REM"], based on physiological signals collected from wearable sensors.',
        "class": {
            "W": "Wakefulness. This includes periods before sleep onset or after final awakening, and short awakenings during the night.",
            "N1": "Stage N1 (light sleep). This is the transition stage between wakefulness and deeper sleep.",
            "N2": "Stage N2 (intermediate sleep). This stage typically follows N1 and occurs multiple times throughout the night.",
            "N3": "Stage N3 (deep sleep). This stage is associated with the most restful sleep and occurs mostly in the first half of the night.",
            "REM": "Rapid Eye Movement (REM) sleep. This stage is associated with dreaming and typically occurs cyclically later in the night.",
        },
        "data": (
            "Data were recorded during overnight sleep using multiple sensors. \n"
            "EEG signals were recorded from two electrode pairs: Fpz-Cz and Pz-Oz, and horizontal EOG was recorded. "
            "Both EEG and EOG signals (in micro volt) were sampled at 100 Hz.\n"
            "Submental EMG was electronically high-pass filtered, rectified, and low-pass filtered to obtain the EMG envelope, "
            "which was expressed as a root-mean-square (RMS) value in micro volt and sampled at 1 Hz.\n"
            "Oro-nasal respiration was also sampled at 1 Hz. "
            "All signals were time-synchronized, and features were computed using a fixed 30-second window.\n"
            "Each feature is named using the format 'modality_featurename', where modality refers to the type of signal "
            "(e.g., EEG-Fpz-Cz, EOG-horizontal) and featurename "
            "describes the extracted characteristic (e.g., mean, std)."
        ),
        "modality_data": {
            "EEG-Fpz-Cz": (
                "EEG data (in micro volt) was collected from the Fpz-Cz electrode pair during sleep and sampled at 100 Hz. "
                "Features were computed from the most recent 30-second window. "
                "Each feature is named using the format 'position_modality_featurename', where featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "EEG-Pz-Oz": (
                "EEG data (in micro volt) was collected from the Pz-Oz electrode pair during sleep and sampled at 100 Hz. "
                "Features were computed from the most recent 30-second window. "
                "Each feature is named using the format 'modality_featurename', where featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "EOG-horizontal": (
                "Horizontal EOG data (in micro volt) was collected to capture eye movements during sleep and sampled at 100 Hz. "
                "Features were computed from the most recent 30-second window. "
                "Each feature is named using the format 'modality_featurename', where featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "EMG-submental": (
                "Submental EMG was high-pass filtered, rectified, and low-pass filtered to compute the envelope, "
                "expressed as an RMS value in micro volt and sampled at 1 Hz. "
                "Features were computed from the most recent 30-second window. "
                "Each feature is named using the format 'modality_featurename', where featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "Resp-oro-nasal": (
                "Oro-nasal respiration was recorded during sleep at 1 Hz in arbitrary units to reflect relative airflow. "
                "Features were computed from the most recent 30-second window. "
                "Each feature is named using the format 'modality_featurename', where featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
        },
        "feature": {
            "EEG-Fpz-Cz": (
                "For EEG signals, a bandpass filter was applied to extract key frequency components: delta (0.5-4 Hz), "
                "theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz), spindle (12-14 Hz), k-complex (0.5-1.5 Hz), and sawtooth (2-6 Hz). "
                "For each band, time-domain features such as mean, standard deviation (std), variance, dynamic range, number of peaks, "
                "number of zero-crossings, and variance of the first-order difference were extracted.\n"
                "Additionally, power spectral density was estimated using Welch's method to compute absolute power for each band. "
                "Ratio features such as delta/theta, theta/alpha, alpha/beta, and (delta+theta)/(alpha+beta) were also included."
            ),
            "EEG-Pz-Oz": (
                "For EEG signals, a bandpass filter was applied to extract key frequency components: delta (0.5-4 Hz), "
                "theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz), spindle (12-14 Hz), k-complex (0.5-1.5 Hz), and sawtooth (2-6 Hz). "
                "For each band, time-domain features such as mean, standard deviation (std), variance, dynamic range, number of peaks, "
                "number of zero-crossings, and variance of the first-order difference were extracted. "
                "Additionally, power spectral density was estimated using Welch's method to compute absolute power for each band. "
                "Ratio features such as delta/theta, theta/alpha, alpha/beta, and (delta+theta)/(alpha+beta) were also included."
            ),
            "EOG-horizontal": (
                "For EOG signals, time-domain features were computed, "
                "including mean, std, variance, dynamic range, number of peaks, "
                "number of zero-crossings, and variance of the first-order difference.\n"
                "The number of large eye movements was detected based on the peak-to-peak amplitude (>120 micro volt) "
                "within a short interval (<=1.5 sec). A separate difference variance was computed after removing these large movements. "
                "Power spectral density was computed using Welch's method. The ratio of slow movement power (0.5–2 Hz) and rapid movement power (2–5 Hz) "
                "to the total EOG power (0.5-30 Hz) were extracted."
            ),
            "EMG-submental": (
                "For EMG signals, time-domain features were computed from the cleaned signal, "
                "including mean, standard deviation (std), dynamic range, absolute integral, median, "
                "10th percentile, and 90th percentile. "
                "Additionally, the number of peaks, mean and std of peak amplitudes, "
                "sum of peak amplitudes, and normalized sum of peak amplitudes were computed."
            ),
            "Resp-oro-nasal": (
                "A bandpass filter with cutoff frequencies of 0.1 Hz and 0.35 Hz was applied to the raw respiration signal. "
                "A peak detection algorithm was then used to identify minima and maxima, allowing estimation of inhalation and exhalation durations. "
                "From these, the mean and standard deviation of inhale and exhale durations were computed, along with their ratio (inhale-to-exhale). "
                "Additional features include stretch (difference between maximum and minimum respiratory signal), inspiration volume (area under the inhalation curve), "
                "respiration rate (mean number of breaths per minute), and average respiration cycle duration."
            ),
        },
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def lowpass_filter(data, cutoff=50, fs=1000, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data)


def count_large_eye_movements(
    eog_cleaned,
    sr,
    amp_thresh=120,
    time_thresh=1.5,
):
    time_limit = int(time_thresh * sr)
    peaks, _ = signal.find_peaks(eog_cleaned)
    troughs, _ = signal.find_peaks(-eog_cleaned)
    events = np.sort(np.concatenate((peaks, troughs)))
    count = 0
    for i in range(len(events) - 1):
        t1, t2 = events[i], events[i + 1]
        if abs(t2 - t1) <= time_limit:
            amp = abs(eog_cleaned[t2] - eog_cleaned[t1])
            if amp >= amp_thresh:
                count += 1
    return count


def remove_large_eye_movements(
    eog_cleaned,
    fs=100,
    amp_thresh=120,
    time_thresh=1.5,
    pad=0.75,
):
    time_limit = int(time_thresh * fs)
    pad_samples = int(pad * fs)
    peaks, _ = signal.find_peaks(eog_cleaned)
    troughs, _ = signal.find_peaks(-eog_cleaned)
    events = np.sort(np.concatenate((peaks, troughs)))
    mask = np.ones_like(eog_cleaned, dtype=bool)
    for i in range(len(events) - 1):
        t1, t2 = events[i], events[i + 1]
        if abs(t2 - t1) <= time_limit:
            amp = abs(eog_cleaned[t2] - eog_cleaned[t1])
            if amp >= amp_thresh:
                start = max(0, min(t1, t2) - pad_samples)
                end = min(len(eog_cleaned), max(t1, t2) + pad_samples)
                mask[start:end] = False
    cleaned_signal = eog_cleaned.copy()
    cleaned_signal[~mask] = np.mean(eog_cleaned[mask])
    return cleaned_signal


def bandpass_filter(data, fs, band, order=4):
    nyq = 0.5 * fs
    low = band[0] / nyq
    high = band[1] / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data)


def process_by_mod(modality, data):
    sr = SAMPLING_RATE
    features = {}
    modality = "-".join(modality.split())
    if "EEG" in modality:
        band_data = data
        signal_modality = modality
        mean_val = np.mean(band_data)
        std_val = np.std(band_data)
        var_val = np.var(band_data)
        features[f"{signal_modality}_mean"] = mean_val
        features[f"{signal_modality}_std"] = std_val
        features[f"{signal_modality}_variance"] = var_val
        dynamic_range = np.max(band_data) - np.min(band_data)
        features[f"{signal_modality}_dynamic_range"] = dynamic_range
        peaks = signal.find_peaks(band_data - mean_val, height=3 * std_val)[0]
        features[f"{signal_modality}_num_peaks"] = len(peaks)
        zero_crossings = np.where(np.diff(np.sign(band_data - mean_val)))[0]
        features[f"{signal_modality}_num_zero_crossings"] = len(zero_crossings)
        differences = band_data[1:] - band_data[:-1]
        difference_variance = np.var(differences)
        features[f"{signal_modality}_difference_variance"] = difference_variance

        freqs, psd = signal.welch(data, fs=sr, nperseg=sr * 2)
        delta_idx = np.logical_and(freqs >= 0.5, freqs <= 4)
        theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
        alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)
        beta_idx = np.logical_and(freqs >= 12, freqs <= 30)
        spindle_idx = np.logical_and(freqs >= 12, freqs <= 14)
        kcomplex_idx = np.logical_and(freqs >= 0.5, freqs <= 1.5)
        sawtooth_idx = np.logical_and(freqs >= 2, freqs <= 6)

        delta_power = np.trapezoid(psd[delta_idx], freqs[delta_idx])
        theta_power = np.trapezoid(psd[theta_idx], freqs[theta_idx])
        alpha_power = np.trapezoid(psd[alpha_idx], freqs[alpha_idx])
        beta_power = np.trapezoid(psd[beta_idx], freqs[beta_idx])
        spindle_power = np.trapezoid(psd[spindle_idx], freqs[spindle_idx])
        kcomplex_power = np.trapezoid(psd[kcomplex_idx], freqs[kcomplex_idx])
        sawtooth_power = np.trapezoid(psd[sawtooth_idx], freqs[sawtooth_idx])

        delta_theta_ratio = delta_power / theta_power if theta_power > 0 else 0
        theta_alpha_ratio = theta_power / alpha_power if alpha_power > 0 else 0
        alpha_beta_ratio = alpha_power / beta_power if beta_power > 0 else 0
        slow_fast_ratio = (
            (delta_power + theta_power) / (alpha_power + beta_power)
            if (alpha_power + beta_power) > 0
            else 0
        )

        features[f"{modality}_delta_power"] = delta_power
        features[f"{modality}_theta_power"] = theta_power
        features[f"{modality}_alpha_power"] = alpha_power
        features[f"{modality}_beta_power"] = beta_power
        features[f"{modality}_spindle_power"] = spindle_power
        features[f"{modality}_kcomplex_power"] = kcomplex_power
        features[f"{modality}_sawtooth_power"] = sawtooth_power
        features[f"{modality}_delta/theta_ratio"] = delta_theta_ratio
        features[f"{modality}_theta/alpha_ratio"] = theta_alpha_ratio
        features[f"{modality}_alpha/beta_ratio"] = alpha_beta_ratio
        features[f"{modality}_(delta+theta)/(alpha+beta)_ratio"] = slow_fast_ratio
    elif "EOG" in modality:
        eog_cleaned = data
        try:
            eog_cleaned = nk.eog_clean(data, sr)
        except IndexError as e:
            features[f"{modality}_mean"] = 0
            features[f"{modality}_std"] = 0
            features[f"{modality}_variance"] = 0
            features[f"{modality}_dynamic_range"] = 0
            features[f"{modality}_num_peaks"] = 0
            features[f"{modality}_num_zero_crossings"] = 0
            features[f"{modality}_difference_variance"] = 0
            features[f"{modality}_num_large_eye_movements"] = 0
            features[f"{modality}_difference_variance_without_large_movements"] = 0
            features[f"{modality}_slow_movement_power_ratio"] = 0
            features[f"{modality}_rapid_movement_power_ratio"] = 0
            return features
        eog_mean = np.mean(eog_cleaned)
        eog_std = np.std(eog_cleaned)
        eog_var = np.var(eog_cleaned)
        features[f"{modality}_mean"] = eog_mean
        features[f"{modality}_std"] = eog_std
        features[f"{modality}_variance"] = eog_var
        dynamic_range = np.max(eog_cleaned) - np.min(eog_cleaned)
        features[f"{modality}_dynamic_range"] = dynamic_range
        peaks = signal.find_peaks(eog_cleaned - eog_mean, height=3 * eog_std)[0]
        features[f"{modality}_num_peaks"] = len(peaks)
        zero_crossings = np.where(np.diff(np.sign(eog_cleaned - eog_mean)))[0]
        features[f"{modality}_num_zero_crossings"] = len(zero_crossings)
        differences = eog_cleaned[1:] - eog_cleaned[:-1]
        difference_variance = np.var(differences)
        features[f"{modality}_difference_variance"] = difference_variance
        features[f"{modality}_num_large_eye_movements"] = count_large_eye_movements(
            eog_cleaned, sr, amp_thresh=120, time_thresh=1.5
        )
        eog_large_movement_removed = remove_large_eye_movements(
            eog_cleaned, fs=sr, amp_thresh=120, time_thresh=1.5, pad=0.75
        )
        differences = eog_large_movement_removed[1:] - eog_large_movement_removed[:-1]
        difference_variance = np.var(differences)
        features[f"{modality}_difference_variance_without_large_movements"] = (
            difference_variance
        )
        freqs, psd = signal.welch(eog_cleaned, fs=sr, nperseg=sr * 2)
        total_idx = np.logical_and(freqs >= 0.5, freqs <= 30)
        total_power = np.trapezoid(psd[total_idx], freqs[total_idx])
        slow_idx = np.logical_and(freqs >= 0.5, freqs <= 2)
        rapid_idx = np.logical_and(freqs >= 2, freqs <= 5)
        slow_power = np.trapezoid(psd[slow_idx], freqs[slow_idx])
        rapid_power = np.trapezoid(psd[rapid_idx], freqs[rapid_idx])
        slow_power_ratio = slow_power / total_power if total_power > 0 else 0
        rapid_power_ratio = rapid_power / total_power if total_power > 0 else 0
        features[f"{modality}_slow_movement_power_ratio"] = slow_power_ratio
        features[f"{modality}_rapid_movement_power_ratio"] = rapid_power_ratio
    elif "Resp" in modality:
        rsp_signals = data
        try:
            rsp_signals, _ = nk.rsp_process(data, sampling_rate=sr, method="biosppy")
        except IndexError as e:
            features[f"{modality}_inhale_duration_mean"] = 0
            features[f"{modality}_inhale_duration_std"] = 0
            features[f"{modality}_exhale_duration_mean"] = 0
            features[f"{modality}_exhale_duration_std"] = 0
            features[f"{modality}_inhale_exhale_ratio"] = 0
            features[f"{modality}_stretch"] = 0
            features[f"{modality}_inspiration_volume"] = 0
            features[f"{modality}_respiration_rate"] = 0
            features[f"{modality}_respiration_duration"] = 0
            return features
        clean = rsp_signals["RSP_Clean"]
        phase = rsp_signals["RSP_Phase"]
        rate = rsp_signals["RSP_Rate"]
        amplitude = rsp_signals["RSP_Amplitude"]
        peaks = np.where(rsp_signals["RSP_Peaks"] == 1)[0]
        troughs = np.where(rsp_signals["RSP_Troughs"] == 1)[0]
        inhale_durations = []
        for t in troughs:
            next_peaks = peaks[peaks > t]
            if len(next_peaks) == 0:
                continue
            inhale_durations.append((next_peaks[0] - t) / sr)
        inhale_durations = np.array(inhale_durations)
        exhale_durations = []
        for p in peaks:
            next_troughs = troughs[troughs > p]
            if len(next_troughs) == 0:
                continue
            exhale_durations.append((next_troughs[0] - p) / sr)
        exhale_durations = np.array(exhale_durations)
        features[f"{modality}_inhale_duration_mean"] = np.mean(inhale_durations)
        features[f"{modality}_inhale_duration_std"] = np.std(inhale_durations)
        features[f"{modality}_exhale_duration_mean"] = np.mean(exhale_durations)
        features[f"{modality}_exhale_duration_std"] = np.std(exhale_durations)
        features[f"{modality}_inhale_exhale_ratio"] = (
            np.mean(inhale_durations) / np.mean(exhale_durations)
            if np.mean(exhale_durations) > 0
            else np.nan
        )
        features[f"{modality}_stretch"] = np.max(clean) - np.min(clean)
        inhale_mask = phase == 1
        features[f"{modality}_inspiration_volume"] = np.trapezoid(
            amplitude[inhale_mask], dx=1 / sr
        )
        features[f"{modality}_respiration_rate"] = np.mean(rate)
        resp_durations = np.diff(troughs) / sr
        features[f"{modality}_respiration_duration"] = np.mean(resp_durations)
    elif "EMG" in modality:
        emg_mean = np.mean(data)
        emg_std = np.std(data)
        features[f"{modality}_mean"] = emg_mean
        features[f"{modality}_std"] = emg_std
        features[f"{modality}_dynamic_range"] = np.max(data) - np.min(data)
        features[f"{modality}_absolute_integral"] = np.sum(np.abs(data)) / sr
        features[f"{modality}_median"] = np.median(data)
        features[f"{modality}_10th_percentile"] = np.percentile(data, 10)
        features[f"{modality}_90th_percentile"] = np.percentile(data, 90)

        peaks, _ = signal.find_peaks(data, height=3 * emg_std)
        peak_values = data[peaks]
        features[f"{modality}_num_peaks"] = len(peaks)
        features[f"{modality}_peak_amplitude_mean"] = (
            np.mean(peak_values) if len(peak_values) > 0 else 0
        )
        features[f"{modality}_peak_amplitude_std"] = (
            np.std(peak_values) if len(peak_values) > 0 else 0
        )
        features[f"{modality}_peak_amplitude_sum"] = (
            np.sum(peak_values) if len(peak_values) > 0 else 0
        )
        features[f"{modality}_peak_amplitude_norm_sum"] = (
            np.sum(peak_values) / np.sum(np.abs(data))
            if np.sum(np.abs(data)) > 0
            else 0
        )
    return features


def preprocess(file_path, missing_ratio=None):
    psg_file_path = file_path[0]
    ann_file_path = file_path[1]
    raw = read_raw_edf(psg_file_path, preload=True, stim_channel=None)
    sr = raw.info["sfreq"]
    curr_user_id = psg_file_path.split("/")[-1].split("-")[0][3:5]

    raw_ch_df = raw.to_data_frame()
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))
    raw_ch_df.drop(columns=["time", "Temp rectal", "Event marker"], inplace=True)
    column_list = raw_ch_df.columns.tolist()
    column_list = [c.replace("_", "-") for c in column_list]

    f = open(psg_file_path, "r", encoding="iso-8859-1")
    reader_raw = BaseEDFReader(f)
    reader_raw.read_header()
    h_raw = reader_raw.header
    f.close()
    raw_start_dt = datetime.strptime(h_raw["date_time"], "%Y-%m-%d %H:%M:%S")
    f = open(ann_file_path, "r", encoding="iso-8859-1")
    reader_ann = BaseEDFReader(f)
    reader_ann.read_header()
    h_ann = reader_ann.header
    _, _, ann = list(zip(*reader_ann.records()))
    f.close()
    ann_start_dt = datetime.strptime(h_ann["date_time"], "%Y-%m-%d %H:%M:%S")
    assert raw_start_dt == ann_start_dt

    remove_idx = []
    labels = []
    label_idx = []
    for a in ann[0]:
        onset_sec, duration_sec, ann_char = a
        ann_str = "".join(ann_char)
        ann_str = ann_str.strip("b'\"")
        label = ann2label[ann_str]
        if label != UNKNOWN:
            if duration_sec % EPOCH_SEC_SIZE != 0:
                assert False, "Duration should be multiple of 30 seconds"
            duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
            label_epoch = np.ones(duration_epoch, dtype=int) * label
            labels.append(label_epoch)
            idx = int(onset_sec * sr) + np.arange(duration_sec * sr, dtype=int)
            label_idx.append(idx)
        else:
            idx = int(onset_sec * sr) + np.arange(duration_sec * sr, dtype=int)
            remove_idx.append(idx)
    labels = np.hstack(labels)

    if len(remove_idx) > 0:
        remove_idx = np.hstack(remove_idx)
        select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
    else:
        select_idx = np.arange(len(raw_ch_df))
    label_idx = np.hstack(label_idx)
    select_idx = np.intersect1d(select_idx, label_idx)

    if len(label_idx) > len(select_idx):
        extra_idx = np.setdiff1d(label_idx, select_idx)
        if np.all(extra_idx > select_idx[-1]):
            n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sr)
            n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * sr)))
            select_idx = select_idx[:-n_trims]
            labels = labels[:-n_label_trims]
        if len(labels) == 0:
            print("No labels left after removing extra labels.")
            return []

    raw_ch = raw_ch_df.values[select_idx]
    if len(raw_ch) % (EPOCH_SEC_SIZE * sr) != 0:
        assert False, "Data length is not multiple of 30 seconds."
    n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sr)

    x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
    y = labels.astype(np.int32)
    assert len(x) == len(y)

    w_edge_mins = 30
    nw_idx = np.where(y != W)[0]
    start_idx = nw_idx[0] - (w_edge_mins * 2)
    end_idx = nw_idx[-1] + (w_edge_mins * 2)
    if start_idx < 0:
        start_idx = 0
    if end_idx >= len(y):
        end_idx = len(y) - 1
    select_idx = np.arange(start_idx, end_idx + 1)
    x = x[select_idx]
    y = y[select_idx]

    num_samples = x.shape[0]
    num_modalities = x.shape[2]
    data = []
    pbar = tqdm(total=num_samples, desc=f"Processing {curr_user_id}", leave=False)
    for i in range(num_samples):
        pbar.update(1)
        if y[i] == 5:
            assert 0
        w_features = {}
        flag = False
        for mod in range(num_modalities):
            mod_data = x[i, :, mod]
            if missing_ratio is not None:
                if np.random.rand() <= missing_ratio:
                    mod_data = np.zeros_like(mod_data)
            mod_features = process_by_mod(column_list[mod], mod_data)
            if mod_features is None:
                assert 0
            for k, v in mod_features.items():
                w_features[k] = v
        if flag:
            continue
        data.append(
            dict(
                user_id=curr_user_id,
                label=class_dict[y[i]],
                features=w_features,
                sample_id=i,
            )
        )
    pbar.close()
    return data


def run(
    path=SLEEPEDF_PATH,
    out_dir=OUT_DIR,
    num_workers=96,
    seed=0,
    missing_ratio=None,
):
    np.random.seed(seed)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    info_path = os.path.join(out_dir, "info.json")
    store_info(info_path)
    print(f"Saved info to {info_path}")
    psg_file_paths = glob(os.path.join(path, "*PSG.edf"))
    ann_file_paths = glob(os.path.join(path, "*Hypnogram.edf"))
    psg_file_paths.sort()
    ann_file_paths.sort()
    file_paths = list(zip(psg_file_paths, ann_file_paths))
    filtered_2013_file_paths = []
    for file_path in file_paths:
        basename = os.path.basename(file_path[0])
        if basename.startswith("SC40"):
            filtered_2013_file_paths.append(file_path)
        elif basename.startswith("SC41"):
            filtered_2013_file_paths.append(file_path)
    all_data = []
    with Pool(processes=num_workers) as pool:
        worker = partial(preprocess, missing_ratio=missing_ratio)
        for data in pool.imap_unordered(worker, filtered_2013_file_paths):
            all_data.extend(data)
    dataset = Dataset.from_list(all_data)
    # count data per label
    labels = set(dataset["label"])
    for label in labels:
        print(
            f"Label '{label}' count: {dataset.filter(lambda x, label=label: x['label'] == label).num_rows}"
        )
    if missing_ratio is not None:
        out_dir = os.path.join(out_dir, "processed_missing")
    else:
        out_dir = os.path.join(out_dir, "processed")
    dataset.save_to_disk(out_dir)
    print(f"Saved dataset to {out_dir}")


if __name__ == "__main__":
    Fire(run)
