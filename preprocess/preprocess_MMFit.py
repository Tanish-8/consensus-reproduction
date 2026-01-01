import os
import csv
import json
import warnings
import numpy as np

from fire import Fire
from glob import glob
from datasets import Dataset
from multiprocessing import Pool
from scipy.fft import rfft, rfftfreq
from functools import partial

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

MMFIT_PATH = "./mmfit"  # path to the mmfit dataset
OUT_DIR = "./mmfit_output"  # path to the output directory
WINDOW_SIZE = 5
STEP_SIZE = 5
SAMPLING_RATES = {
    "pose-3d": 30,
    "smartwatch-left_accelerometer": 100,
    "smartwatch-right_accelerometer": 100,
    "smartwatch-left_gyroscope": 100,
    "smartwatch-right_gyroscope": 100,
    "smartwatch-left_heart-rate": 1,
    "smartwatch-right_heart-rate": 1,
    "earbud-left_accelerometer": 90,
    "earbud-left_gyroscope": 90,
    "smartphone-right_accelerometer": 210,
    "smartphone-right_gyroscope": 210,
    "smartphone-right_magnetometer": 100,
}


def store_info(info_path):
    info = {
        "task": 'Classify the performed exercise into one of the ten workout categories: ["squats", "pushups", "dumbbell_shoulder_press", "lunges", "dumbbell_rows", "situps", "tricep_extensions", "bicep_curls", "lateral_shoulder_raises", "jumping_jacks"], based on physiological signals collected from wearable sensors.',
        "class": {
            "squats": "The subject performed squats",
            "pushups": "The subject performed pushups",
            "dumbbell_shoulder_press": "The subject performed dumbbell shoulder press",
            "lunges": "The subject performed lunges",
            "dumbbell_rows": "The subject performed dumbbell rows",
            "situps": "The subject performed situps",
            "tricep_extensions": "The subject performed tricep extensions",
            "bicep_curls": "The subject performed bicep curls",
            "lateral_shoulder_raises": "The subject performed lateral shoulder raises",
            "jumping_jacks": "The subject performed jumping jacks",
        },
        "data": (
            "Physiological and motion data were collected using three types of wearable devices:\n"
            "- Earbud: Provided triaxial accelerometer (m/s^2) and gyroscope (rad/s) signals sampled at 90 Hz from the left earbud.\n"
            "- Smartwatches: Recorded triaxial accelerometer (m/s^2) and gyroscope (rad/s) signals at 100 Hz, and heart rate (beats per minute) at 1 Hz from the optical PPG sensor from the left and right wrists.\n"
            "- Smartphone: Recorded triaxial accelerometer (m/s^2) and gyroscope (rad/s) signals at 210 Hz, and magnetometer (μT) readings at 100 Hz. Smartphone was placed in the right pocket.\n\n"
            "All signals were time-synchronized, and features were computed from the most recent segment of data. A 5-second window was used for most modalities.\n"
            "Each feature is named using the format 'device_modality_featurename', where device indicates the sensor location (e.g., earbud-left, smartwatch-right), modality refers to the signal type (e.g., accelerometer, gyroscope), "
            "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
        ),
        "modality_data": {
            "earbud-left_accelerometer": (
                "Triaxial accelerometer (m/s^2) signals were recorded at 90 Hz from the left earbud."
                "Features were computed from the most recent 5-second window."
                "Each feature is named using the format 'device_modality_featurename', where modality refers to the signal type (e.g., accelerometer), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "earbud-left_gyroscope": (
                "Triaxial gyroscope (rad/s) signals were recorded at 90 Hz from the left earbud."
                "Features were computed from the most recent 5-second window."
                "Each feature is named using the format 'device_modality_featurename', where modality refers to the signal type (e.g., gyroscope), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "smartwatch-left_accelerometer": (
                "Triaxial accelerometer (m/s^2) signals were recorded at 100 Hz from the left wrist."
                "Features were computed from the most recent 5-second window."
                "Each feature is named using the format 'device_modality_featurename', where modality refers to the signal type (e.g., accelerometer), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "smartwatch-left_gyroscope": (
                "Triaxial gyroscope (rad/s) signals were recorded at 100 Hz from the left wrist."
                "Features were computed from the most recent 5-second window."
                "Each feature is named using the format 'device_modality_featurename', where modality refers to the signal type (e.g., gyroscope), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "smartwatch-right_accelerometer": (
                "Triaxial accelerometer (m/s^2) signals were recorded at 100 Hz from the right wrist."
                "Features were computed from the most recent 5-second window."
                "Each feature is named using the format 'device_modality_featurename', where modality refers to the signal type (e.g., accelerometer), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "smartwatch-right_gyroscope": (
                "Triaxial gyroscope (rad/s) signals were recorded at 100 Hz from the right wrist."
                "Features were computed from the most recent 5-second window."
                "Each feature is named using the format 'device_modality_featurename', where modality refers to the signal type (e.g., gyroscope), "
                "and featurename describes the extracted statistic or characteristic (e.g., _mean, _std)."
            ),
            "smartwatch-left_heart-rate": (
                "Heart rate (beats per minute) signals were recorded at 1 Hz from the left wrist."
                "Features were computed from the most recent 5-second window."
                "Each feature is named using the format 'device_modality_featurename', where modality refers to the signal type (e.g., heart-rate), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "smartwatch-right_heart-rate": (
                "Heart rate (beats per minute) signals were recorded at 1 Hz from the right wrist."
                "Features were computed from the most recent 5-second window."
                "Each feature is named using the format 'device_modality_featurename', where modality refers to the signal type (e.g., heart-rate), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "smartphone-right_accelerometer": (
                "Triaxial accelerometer (m/s^2) signals were recorded at 210 Hz from the right pocket."
                "Features were computed from the most recent 5-second window."
                "Each feature is named using the format 'device_modality_featurename', where modality refers to the signal type (e.g., accelerometer), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "smartphone-right_gyroscope": (
                "Triaxial gyroscope (rad/s) signals were recorded at 210 Hz from the right pocket."
                "Features were computed from the most recent 5-second window."
                "Each feature is named using the format 'device_modality_featurename', where modality refers to the signal type (e.g., gyroscope), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "smartphone-right_magnetometer": (
                "Magnetometer (μT) signals were recorded at 100 Hz from the right pocket."
                "Features were computed from the most recent 5-second window."
                "Each feature is named using the format 'device_modality_featurename', where modality refers to the signal type (e.g., magnetometer), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
        },
        "feature": {
            "accelerometer": (
                "For raw accelerometer signals, the mean, standard deviation (std), and absolute integral were computed "
                "for each axis (x, y, z) and for the overall magnitude (mag). In addition, the peak frequency was extracted for each axis."
            ),
            "gyroscope": (
                "For raw gyroscope signals, the mean, standard deviation (std), and absolute integral were computed "
                "for each axis (x, y, z) and for the overall magnitude (mag). In addition, the peak frequency was extracted for each axis."
            ),
            "magnetometer": (
                "For raw magnetometer signals, the mean, standard deviation (std), and absolute integral were computed "
                "for each axis (x, y, z) and for the overall magnitude (mag). In addition, the peak frequency was extracted for each axis."
            ),
            "heart-rate": ("For raw heart rate signals, the mean value was computed."),
        },
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def load_modality(filepath):
    """
    Loads modality from filepath and returns numpy array, or None if no file is found.
    :param filepath: File path to MM-Fit modality.
    :return: MM-Fit modality (numpy array).
    """
    try:
        mod = np.load(filepath)
    except FileNotFoundError as e:
        mod = None
        print("{}. Returning None".format(e))
    return mod


def load_labels(filepath):
    """
    Loads and reads CSV MM-Fit CSV label file.
    :param filepath: File path to a MM-Fit CSV label file.
    :return: List of lists containing label data, (Start Frame, End Frame, Repetition Count, Activity) for each
    exercise set.
    """
    labels = []
    with open(filepath, "r") as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            labels.append([int(line[0]), int(line[1]), int(line[2]), line[3]])
    return labels


def process_by_mod(mod_name, mod_data):
    features = {}
    sr = SAMPLING_RATES[mod_name]
    if (
        "accelerometer" in mod_name
        or "gyroscope" in mod_name
        or "magnetometer" in mod_name
    ):
        mean = np.mean(mod_data, axis=0)
        std = np.std(mod_data, axis=0)
        freqs = rfftfreq(mod_data.shape[0], d=1 / sr)
        fft_vals = np.abs(rfft(mod_data - mean, axis=0))
        peak_freq = freqs[np.argmax(fft_vals, axis=0)]
        for i, axis in enumerate(["x", "y", "z"]):
            features[f"{mod_name}_{axis}_mean"] = mean[i]
            features[f"{mod_name}_{axis}_std"] = std[i]
            features[f"{mod_name}_{axis}_peak_frequency"] = peak_freq[i]
            features[f"{mod_name}_{axis}_absolute_integral"] = (
                np.sum(np.abs(mod_data[:, i])) / sr
            )
        mag = np.sqrt(np.sum(np.square(mod_data), axis=1))
        mag_mean = np.mean(mag)
        mag_std = np.std(mag)
        features[f"{mod_name}_mag_mean"] = mag_mean
        features[f"{mod_name}_mag_std"] = mag_std
        features[f"{mod_name}_mag_absolute_integral"] = np.sum(np.abs(mag)) / sr
    elif "heart-rate" in mod_name:
        features[f"{mod_name}_mean"] = np.mean(mod_data)

    return features


def preprocess(user_path, missing_ratio=None):
    user_id = os.path.basename(user_path)
    if user_id == "w20" or user_id == "w12" or user_id == "w04" or user_id == "w07":
        return None
    if (
        not os.path.exists(os.path.join(user_path, f"{user_id}_pose_3d.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_sw_l_acc.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_sw_r_acc.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_sw_l_gyr.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_sw_r_gyr.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_sw_l_hr.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_sw_r_hr.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_eb_l_acc.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_eb_l_gyr.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_sp_r_acc.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_sp_r_gyr.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_sp_r_mag.npy"))
        or not os.path.exists(os.path.join(user_path, f"{user_id}_labels.csv"))
    ):
        return None

    pose_3d_path = os.path.join(user_path, f"{user_id}_pose_3d.npy")
    sw_l_acc_path = os.path.join(user_path, f"{user_id}_sw_l_acc.npy")
    sw_r_acc_path = os.path.join(user_path, f"{user_id}_sw_r_acc.npy")
    sw_l_gyr_path = os.path.join(user_path, f"{user_id}_sw_l_gyr.npy")
    sw_r_gyr_path = os.path.join(user_path, f"{user_id}_sw_r_gyr.npy")
    sw_l_hr_path = os.path.join(user_path, f"{user_id}_sw_l_hr.npy")
    sw_r_hr_path = os.path.join(user_path, f"{user_id}_sw_r_hr.npy")
    eb_l_acc_path = os.path.join(user_path, f"{user_id}_eb_l_acc.npy")
    eb_l_gyr_path = os.path.join(user_path, f"{user_id}_eb_l_gyr.npy")
    sp_r_acc_path = os.path.join(user_path, f"{user_id}_sp_r_acc.npy")
    sp_r_gyr_path = os.path.join(user_path, f"{user_id}_sp_r_gyr.npy")
    sp_r_mag_path = os.path.join(user_path, f"{user_id}_sp_r_mag.npy")
    labels_path = os.path.join(user_path, f"{user_id}_labels.csv")

    modalities = {
        "pose-3d": load_modality(pose_3d_path),
        "smartwatch-left_accelerometer": load_modality(sw_l_acc_path),
        "smartwatch-right_accelerometer": load_modality(sw_r_acc_path),
        "smartwatch-left_gyroscope": load_modality(sw_l_gyr_path),
        "smartwatch-right_gyroscope": load_modality(sw_r_gyr_path),
        "smartwatch-left_heart-rate": load_modality(sw_l_hr_path),
        "smartwatch-right_heart-rate": load_modality(sw_r_hr_path),
        "earbud-left_accelerometer": load_modality(eb_l_acc_path),
        "earbud-left_gyroscope": load_modality(eb_l_gyr_path),
        "smartphone-right_accelerometer": load_modality(sp_r_acc_path),
        "smartphone-right_gyroscope": load_modality(sp_r_gyr_path),
        "smartphone-right_magnetometer": load_modality(sp_r_mag_path),
    }
    labels = load_labels(labels_path)

    window_size = WINDOW_SIZE
    step_size = STEP_SIZE
    time_idx = 0
    data = []
    while time_idx < modalities["pose-3d"].shape[1]:
        start_time = modalities["pose-3d"][0, time_idx, 0]
        w_features = {}
        for modality, mod_data in modalities.items():
            if modality == "pose-3d":
                continue
            start_idx = np.searchsorted(mod_data[:, 0], start_time, "left")
            end_idx = start_idx + int(window_size * SAMPLING_RATES[modality])
            if end_idx >= len(mod_data):
                w_features = {}
                break
            window_data = mod_data[start_idx:end_idx]
            if "heart-rate" in modality:
                window_data = window_data[:, 2]
            else:
                window_data = window_data[:, 2:]
            if missing_ratio is not None:
                if np.random.rand() <= missing_ratio:
                    window_data = np.zeros_like(window_data)
            mod_features = process_by_mod(modality, window_data)
            if mod_features is None:
                w_features = {}
                break
            for k, v in mod_features.items():
                w_features[k] = v
        if len(w_features) == 0:
            time_idx += int(step_size * SAMPLING_RATES["pose-3d"])
            continue

        middle_time = int(start_time + window_size * SAMPLING_RATES["pose-3d"] / 2)
        label = None
        for label_row in labels:
            if label_row[0] <= middle_time <= label_row[1]:
                label = label_row[3]
                break
        if label is None:
            time_idx += int(step_size * SAMPLING_RATES["pose-3d"])
            continue

        data.append(
            dict(
                user_id=user_id,
                label=label,
                features=w_features,
                sample_id=start_time,
            )
        )
        time_idx += int(step_size * SAMPLING_RATES["pose-3d"])
    return data


def run(path=MMFIT_PATH, out_dir=OUT_DIR, num_workers=96, seed=0, missing_ratio=None):
    np.random.seed(seed)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    info_path = os.path.join(out_dir, "info.json")
    store_info(info_path)
    print(f"Saved info to {info_path}")
    users = glob(os.path.join(path, "w*"))
    all_data = []
    with Pool(processes=num_workers) as pool:
        worker = partial(preprocess, missing_ratio=missing_ratio)
        for user_data in pool.imap_unordered(worker, users):
            if not user_data:
                continue
            all_data.extend(user_data)
    dataset = Dataset.from_list(all_data)
    labels = dataset.unique("label")
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
