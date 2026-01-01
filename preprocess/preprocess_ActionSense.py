import os
import h5py
import json
import warnings
import numpy as np
import neurokit2 as nk

from fire import Fire
from glob import glob
from datasets import Dataset
from multiprocessing import Pool
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.fft import rfft, rfftfreq
from functools import partial
from neurokit2.misc._warnings import NeuroKitWarning

warnings.simplefilter("ignore", NeuroKitWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

ACTIONSENSE_PATH = "./actionsense"  # path to the actionsense dataset
OUT_DIR = "./actionsense_output"  # path to the output directory
SLIDING_WINDOW = 5  # seconds
DURATION = 5  # seconds
SAMPLING_RATES = {
    "left-arm_accelerometer": 45,
    "left-arm_angular-velocity": 45,
    "left-arm_emg": 160,
    "right-arm_accelerometer": 45,
    "right-arm_angular-velocity": 45,
    "right-arm_emg": 160,
}


def store_info(info_path):
    info = {
        "task": 'Classify the user\'s kitchen activity: ["spreading", "peeling or slicing", "wiping", "opening or closing a jar", "tableware task"], based on physiological signals collected from wearable sensors.',
        "class": {
            "peeling or slicing": "When peeling, the user peels a cucumber or potato using the peeler. When slicing, the user slices a cucumber, potato, or bread using a chef's knife. Slicing cross-sectionally to create rings is preferred.",
            "spreading": "The user uses a dinner knife to spread almond butter or jelly onto a slice of bread. The goal is to fully cover the slice, but the amount of spread and the technique may vary. For example, the user may place the slice on a table or hold it in their hand during the process.",
            "wiping": "The user cleans a plate or pan by wiping away dirt using a sponge or towel.",
            "opening and closing a jar": "The user opens or closes a jar of almond butter.",
            "tableware task": "Tableware tasks include setting the table, stacking tableware, loading the dishwasher, and unloading it. When setting the table, the user places plates on the table. For stacking, the user organizes large plates, small plates, and bowls in preparation for loading. During loading, the user places plates, bowls, mugs, glasses, knives, forks, and spoons into the dishwasher. Unloading involves removing these items from the dishwasher and placing them in their designated storage locations.",
        },
        "data": (
            "Three sensors were attached to each forearm. \n"
            "Electromyography (EMG) (in mV) was recorded using a dry electrode at 160 Hz, "
            "while accelerometer (in g) and angular velocity (in degrees per second) "
            "were both recorded at 45 Hz.\n"
            "All signals were time-synchronized, and features were computed from the most recent segment of data. A 5-second window was used for most modalities.\n"
            "Each feature is named using the format 'position_modality_featurename', where position indicates the sensor location (e.g., left-arm, right-arm), modality refers to the signal type (e.g., accelerometer, emg), and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
        ),
        "modality_data": {
            "left-arm_accelerometer": (
                "Accelerometer data was collected from the left forearm (in g) along three axes (x, y, z) and the overall magnitude, sampled at 45 Hz. "
                "Note that the sensor orientation is unknown, so the axes may not correspond to the physical axes. "
                "Features were computed from the most recent 5-second window. "
                "Each feature is named using the format 'position_modality_featurename', where featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "left-arm_angular-velocity": (
                "Angular velocity data was collected from the left forearm (in degrees per second) along three axes (x, y, z) and the overall magnitude, sampled at 45 Hz. "
                "Note that the sensor orientation is unknown, so the axes may not correspond to the physical axes. "
                "Features were computed from the most recent 5-second window. "
                "Each feature is named using the format 'position_modality_featurename', where featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "left-arm_emg": (
                "Electromyogram data was collected from the left forearm (in mV), sampled at 160 Hz. "
                "Features were computed from the most recent 5-second window. "
                "Each feature is named using the format 'position_modality_featurename', where featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "right-arm_accelerometer": (
                "Accelerometer data was collected from the right forearm (in g) along three axes (x, y, z) and the overall magnitude, sampled at 45 Hz. "
                "Note that the sensor orientation is unknown, so the axes may not correspond to the physical axes. "
                "Features were computed from the most recent 5-second window. "
                "Each feature is named using the format 'position_modality_featurename', where featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "right-arm_angular-velocity": (
                "Angular velocity data was collected from the right forearm (in degrees per second) along three axes (x, y, z) and the overall magnitude, sampled at 45 Hz. "
                "Note that the sensor orientation is unknown, so the axes may not correspond to the physical axes. "
                "Features were computed from the most recent 5-second window. "
                "Each feature is named using the format 'position_modality_featurename', where featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "right-arm_emg": (
                "Electromyogram data was collected from the right forearm (in mV), sampled at 160 Hz. "
                "Features were computed from the most recent 5-second window. "
                "Each feature is named using the format 'position_modality_featurename', where featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
        },
        "feature": {
            "accelerometer": (
                "For raw accelerometer signals, the mean, standard deviation (std), and absolute integral were computed "
                "for each axis (x, y, z) and for the overall magnitude (mag). In addition, the peak frequency was extracted for each axis."
            ),
            "angular-velocity": (
                "For raw angular velocity signals, the mean, standard deviation (std), and absolute integral were computed "
                "for each axis (x, y, z) and for the overall magnitude (mag). In addition, the peak frequency was extracted for each axis."
            ),
            "emg": (
                "A high-pass filter was applied to remove the DC component. From the resulting signal, features were computed: "
                "mean, standard deviation, dynamic range, absolute integral, median, 10th percentile, 90th percentile, and frequency-domain features including mean, median, and peak frequency. "
                "The number of peaks, mean and standard deviation of peak amplitudes, sum of peak amplitudes, and normalized sum of peak amplitudes were computed."
            ),
        },
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def process_by_mod(modality, data):
    sr = SAMPLING_RATES[modality]
    features = {}
    if "accelerometer" in modality or "angular-velocity" in modality:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        freqs = rfftfreq(data.shape[0], d=1 / sr)
        fft_vals = np.abs(rfft(data - mean, axis=0))
        peak_freq = freqs[np.argmax(fft_vals, axis=0)]
        for i, axis in enumerate(["x", "y", "z"]):
            features[f"{modality}_{axis}_mean"] = mean[i]
            features[f"{modality}_{axis}_std"] = std[i]
            features[f"{modality}_{axis}_peak_frequency"] = peak_freq[i]
            features[f"{modality}_{axis}_absolute_integral"] = (
                np.sum(np.abs(data[:, i])) / sr
            )
        mag = np.sqrt(np.sum(np.square(data), axis=1))
        mag_mean = np.mean(mag)
        mag_std = np.std(mag)
        features[f"{modality}_mag_mean"] = mag_mean
        features[f"{modality}_mag_std"] = mag_std
        features[f"{modality}_mag_absolute_integral"] = np.sum(np.abs(mag)) / sr

    elif "emg" in modality:
        data = data[:, 0]
        emg_cleaned = nk.emg_clean(data, sampling_rate=sr, method="none")
        features[f"{modality}_mean"] = np.mean(emg_cleaned)
        features[f"{modality}_std"] = np.std(emg_cleaned)
        max_val = np.max(emg_cleaned)
        min_val = np.min(emg_cleaned)
        features[f"{modality}_dynamic_range"] = max_val - min_val
        features[f"{modality}_absolute_integral"] = np.sum(np.abs(emg_cleaned)) / sr
        features[f"{modality}_median"] = np.median(emg_cleaned)
        features[f"{modality}_10th_percentile"] = np.percentile(emg_cleaned, 10)
        features[f"{modality}_90th_percentile"] = np.percentile(emg_cleaned, 90)

        frequencies, power = welch(emg_cleaned, fs=sr, nperseg=sr * 2)
        power_norm = power / np.sum(power)
        features[f"{modality}_mean_freq"] = np.sum(frequencies * power_norm)
        ps = np.cumsum(power_norm)
        if len(np.where(ps >= 0.5)[0]) == 0:
            features[f"{modality}_median_freq"] = 0
        else:
            features[f"{modality}_median_freq"] = frequencies[np.where(ps >= 0.5)[0][0]]
        features[f"{modality}_peak_freq"] = frequencies[np.argmax(power)]

        peaks, _ = find_peaks(emg_cleaned, height=np.std(emg_cleaned))
        pv = emg_cleaned[peaks]
        features[f"{modality}_num_peaks"] = len(peaks)
        features[f"{modality}_peak_amplitude_mean"] = np.mean(pv) if len(pv) > 0 else 0
        features[f"{modality}_peak_amplitude_std"] = np.std(pv) if len(pv) > 0 else 0
        features[f"{modality}_peak_amplitude_sum"] = np.sum(pv) if len(pv) > 0 else 0
        features[f"{modality}_peak_amplitude_norm_sum"] = (
            np.sum(pv) / np.sum(np.abs(emg_cleaned))
            if np.sum(np.abs(emg_cleaned)) > 0
            else 0
        )

    return features


def get_labels(h5_data):
    device_name = "experiment-activities"
    stream_name = "activities"

    activity_datas = h5_data[device_name][stream_name]["data"]
    activity_times_s = h5_data[device_name][stream_name]["time_s"]
    activity_times_s = np.squeeze(np.array(activity_times_s))
    activity_datas = [[x.decode("utf-8") for x in datas] for datas in activity_datas]

    exclude_bad_labels = True
    activities_labels = []
    activities_start_times_s = []
    activities_end_times_s = []
    activities_ratings = []
    activities_notes = []
    for row_index, time_s in enumerate(activity_times_s):
        label = activity_datas[row_index][0].lower()
        if label.startswith("spread"):
            label = "spreading"
        elif label.startswith("peel") or label.startswith("slice"):
            label = "peeling or slicing"
        elif label.startswith("open"):
            label = "opening and closing a jar"
        elif label.startswith("clean"):
            label = "wiping"
        elif (
            label.startswith("set")
            or label.startswith("stack")
            or label.startswith("load")
            or label.startswith("unload")
        ):
            label = "tableware task"
        else:
            continue
        is_start = activity_datas[row_index][1] == "Start"
        is_stop = activity_datas[row_index][1] == "Stop"
        rating = activity_datas[row_index][2]
        notes = activity_datas[row_index][3]
        if exclude_bad_labels and rating in ["Bad", "Maybe"]:
            continue

        if is_start:
            activities_labels.append(label)
            activities_start_times_s.append(time_s)
            activities_ratings.append(rating)
            activities_notes.append(notes)

        if is_stop:
            activities_end_times_s.append(time_s)

    return activities_labels, activities_start_times_s, activities_end_times_s


def preprocess(hdf5_file, missing_ratio=None):
    h5_data = h5py.File(hdf5_file, "r")
    user_id = os.path.basename(hdf5_file).split(".")[0].split("_")[-1]
    labels, label_start_times, label_end_times = get_labels(h5_data)

    modality_data = {}
    modality_timestamps = {}
    for device_name in ["myo-left", "myo-right"]:
        for sensor_name in ["acceleration_g", "angular_velocity_deg_s", "emg"]:
            if "left" in device_name:
                modality_prefix = "left-arm"
            else:
                modality_prefix = "right-arm"
            if "acceleration" in sensor_name:
                modality_suffix = "accelerometer"
            elif "angular_velocity" in sensor_name:
                modality_suffix = "angular-velocity"
            else:
                modality_suffix = "emg"
            modality_name = f"{modality_prefix}_{modality_suffix}"
            if modality_name not in modality_data:
                data = np.array(h5_data[device_name][sensor_name]["data"])
                timestamps = np.squeeze(
                    np.array(h5_data[device_name][sensor_name]["time_s"])
                )
                modality_data[modality_name] = data
                modality_timestamps[modality_name] = timestamps

    data = []
    for label, start_time, end_time in zip(labels, label_start_times, label_end_times):
        window_start = start_time
        window_end = start_time + DURATION
        while window_end <= end_time:
            w_features = {}
            skip_window = False
            for modality, mod_data in modality_data.items():
                timestamps = modality_timestamps[modality]
                start_idx = np.where(timestamps >= window_start)[0][0]
                if start_idx == 0:
                    skip_window = True
                    break
                end_idx = start_idx + int(DURATION * SAMPLING_RATES[modality])
                if end_idx >= len(timestamps):
                    skip_window = True
                    break
                if (
                    timestamps[end_idx] < window_end - 1
                    and timestamps[end_idx] > window_end + 1
                ):
                    skip_window = True
                    break

                window_data = mod_data[start_idx:end_idx]
                if missing_ratio is not None:
                    if np.random.rand() <= missing_ratio:
                        window_data = np.zeros_like(window_data)
                mod_features = process_by_mod(modality, window_data)
                if mod_features is None:
                    skip_window = True
                    break
                for k, v in mod_features.items():
                    w_features[k] = v
            if skip_window:
                window_start += SLIDING_WINDOW
                window_end += SLIDING_WINDOW
                continue
            data.append(
                dict(
                    user_id=user_id,
                    label=label,
                    features=w_features,
                    sample_id=window_start,
                )
            )
            window_start += SLIDING_WINDOW
            window_end += SLIDING_WINDOW

    return data


def run(
    path=ACTIONSENSE_PATH,
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
    hdf5_files = glob(os.path.join(path, "*.hdf5"))
    all_data = []
    with Pool(processes=num_workers) as pool:
        worker = partial(preprocess, missing_ratio=missing_ratio)
        for data in pool.imap_unordered(worker, hdf5_files):
            all_data.extend(data)
    dataset = Dataset.from_list(all_data)
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
