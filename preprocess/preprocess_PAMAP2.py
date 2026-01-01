import os
import json
import warnings
import numpy as np
import pandas as pd

from fire import Fire
from glob import glob
from datasets import Dataset
from multiprocessing import Pool
from scipy.fft import rfft, rfftfreq
from functools import partial

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

PAMAP2_PATH = "./pamap2"  # path to the pamap2 dataset
OUT_DIR = "./pamap2_output"  # path to the output directory
SLIDING_WINDOW = 5.12  # seconds
SAMPLING_RATE = 100  # Hz
LABEL_DICT = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic walking",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    24: "rope jumping",
}


def store_info(info_path):
    info = {
        "task": 'Classify the user\'s activity into one of the following labels: ["lying", "sitting", "standing", "walking", "running", "cycling", "nordic walking", "ascending stairs", "descending stairs", "vacuum cleaning", "ironing", "rope jumping"].',
        "class": {
            "lying": "Lying quietly while doing nothing. Small movements such as changing the lying posture are allowed.",
            "sitting": "Sitting in a chair in whatever posture the subject feels comfortable. Changing sitting postures is also allowed.",
            "standing": "Standing still or standing still and talking, possibly gesticulating.",
            "ironing": "Ironing 1-2 shirts or T-shirts.",
            "vacuuming": "Vacuum cleaning one or two office rooms (which includes moving objects, e.g. chairs, placed on the floor).",
            "ascending stairs": "Performed in a building between the ground and the top floors, a distance of five floors had to be covered going upstairs.",
            "descending stairs": "Performed in a building between the ground and the top floors, a distance of five floors had to be covered going downstairs.",
            "normal walking": "Walking outside with moderate to brisk pace with a speed of 4-6km/h, according to what was suitable for the subject.",
            "nordic walking": "Performed outside on asphaltic terrain, using asphalt pads on the walking poles (it has to be noted, that none of the subjects was very familiar with this sport activity).",
            "cycling": "Performed outside with a real bike with slow to moderate pace, as if the subject would bike to work or bike for pleasure (but not as a sport activity).",
            "running": "Meant jogging outside with a suitable speed for the individual subjects.",
            "rope jumping": "The subjects used the technique most suitable for them, which mainly consisted of the basic jump (where both feet jump at the same time over the rope) or the alternate foot jump (where alternate feet are used to jump off the ground).",
        },
        "data": (
            "Three IMU sensors were attached to the chest, hand, and ankle of the subject.\n"
            "In each sensor, accelerometer (in m/s^2) and gyroscope (in rad/s) data were recorded at 100 Hz.\n"
            "All signals were time-synchronized, and features were computed from the most recent 5.12 seconds of data.\n"
            "Each feature is named using the format 'position_modality_featurename', "
            "where position indicates the sensor location (e.g., chest, hand, ankle), "
            "modality refers to the signal type (e.g., accelerometer, gyroscope), "
            "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
        ),
        "modality_data": {
            "hand_accelerometer": (
                "Accelerometer data was collected from the hand (in m/s^2) along three axes (x, y, z) and the overall magnitude, sampled at 100 Hz. "
                "Features were computed from the most recent 5.12 seconds of data. "
                "Each feature is named using the format 'position_modality_featurename', "
                "where position indicates the sensor location (hand), "
                "modality refers to the signal type (accelerometer), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "hand_gyroscope": (
                "Gyroscope data was collected from the hand (in rad/s) along three axes (x, y, z) and the overall magnitude, sampled at 100 Hz. "
                "Features were computed from the most recent 5.12 seconds of data. "
                "Each feature is named using the format 'position_modality_featurename', "
                "where position indicates the sensor location (hand), "
                "modality refers to the signal type (gyroscope), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "chest_accelerometer": (
                "Accelerometer data was collected from the chest (in m/s^2) along three axes (x, y, z) and the overall magnitude, sampled at 100 Hz. "
                "Features were computed from the most recent 5.12 seconds of data. "
                "Each feature is named using the format 'position_modality_featurename', "
                "where position indicates the sensor location (chest), "
                "modality refers to the signal type (accelerometer), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "chest_gyroscope": (
                "Gyroscope data was collected from the chest (in rad/s) along three axes (x, y, z) and the overall magnitude, sampled at 100 Hz. "
                "Features were computed from the most recent 5.12 seconds of data. "
                "Each feature is named using the format 'position_modality_featurename', "
                "where position indicates the sensor location (chest), "
                "modality refers to the signal type (gyroscope), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "ankle_accelerometer": (
                "Accelerometer data was collected from the ankle (in m/s^2) along three axes (x, y, z) and the overall magnitude, sampled at 100 Hz. "
                "Features were computed from the most recent 5.12 seconds of data. "
                "Each feature is named using the format 'position_modality_featurename', "
                "where position indicates the sensor location (ankle), "
                "modality refers to the signal type (accelerometer), "
                "and featurename describes the extracted statistic or characteristic (e.g., mean, std)."
            ),
            "ankle_gyroscope": (
                "Gyroscope data was collected from the ankle (in rad/s) along three axes (x, y, z) and the overall magnitude, sampled at 100 Hz. "
                "Features were computed from the most recent 5.12 seconds of data. "
                "Each feature is named using the format 'position_modality_featurename', "
                "where position indicates the sensor location (ankle), "
                "modality refers to the signal type (gyroscope), "
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
        },
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def process_by_mod(mod_name, data):
    features = {}
    if "accelerometer" in mod_name or "gyroscope" in mod_name:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        freqs = rfftfreq(data.shape[0], d=1 / SAMPLING_RATE)
        fft_vals = np.abs(rfft(data - mean, axis=0))
        peak_freq = freqs[np.argmax(fft_vals, axis=0)]
        for i, axis in enumerate(["x", "y", "z"]):
            features[f"{mod_name}_{axis}_mean"] = mean[i]
            features[f"{mod_name}_{axis}_std"] = std[i]
            features[f"{mod_name}_{axis}_peak_frequency"] = peak_freq[i]
            features[f"{mod_name}_{axis}_absolute_integral"] = (
                np.sum(np.abs(data[:, i])) / SAMPLING_RATE
            )
        mag = np.sqrt(np.sum(np.square(data), axis=1))
        mag_mean = np.mean(mag)
        mag_std = np.std(mag)
        features[f"{mod_name}_mag_mean"] = mag_mean
        features[f"{mod_name}_mag_std"] = mag_std
        features[f"{mod_name}_mag_absolute_integral"] = (
            np.sum(np.abs(mag)) / SAMPLING_RATE
        )
    return features


def preprocess(user_path, missing_ratio=None):
    user_id = os.path.basename(user_path).split(".")[0]
    raw_data = pd.read_csv(user_path, sep=" ", header=None).values
    window_size = int(SLIDING_WINDOW * SAMPLING_RATE)
    start_idx = 0
    end_idx = start_idx + window_size
    data = []
    while end_idx < raw_data.shape[0]:
        window_data = raw_data[start_idx:end_idx]
        time_start = window_data[0, 0]
        time_end = window_data[-1, 0]
        duration = time_end - time_start
        if duration < SLIDING_WINDOW * 0.9 or duration > SLIDING_WINDOW * 1.1:
            start_idx += window_size
            end_idx += window_size
            continue
        activity = window_data[:, 1]
        if len(np.unique(activity)) != 1:
            start_idx += window_size
            end_idx += window_size
            continue
        activity = int(activity[0])
        if activity not in LABEL_DICT.keys():
            start_idx += window_size
            end_idx += window_size
            continue
        activity = LABEL_DICT[activity]
        acc_hand = window_data[:, 3:20][:, 1:4]
        gyr_hand = window_data[:, 3:20][:, 7:10]
        acc_chest = window_data[:, 20:37][:, 1:4]
        gyr_chest = window_data[:, 20:37][:, 7:10]
        acc_ankle = window_data[:, 37:54][:, 1:4]
        gyr_ankle = window_data[:, 37:54][:, 7:10]
        # if more than 5% of the data is nan, skip
        if (
            np.isnan(acc_hand).sum() > 0.05 * acc_hand.size
            or np.isnan(gyr_hand).sum() > 0.05 * gyr_hand.size
            or np.isnan(acc_chest).sum() > 0.05 * acc_chest.size
            or np.isnan(gyr_chest).sum() > 0.05 * gyr_chest.size
            or np.isnan(acc_ankle).sum() > 0.05 * acc_ankle.size
            or np.isnan(gyr_ankle).sum() > 0.05 * gyr_ankle.size
        ):
            start_idx += window_size
            end_idx += window_size
            continue

        def _fill_missing(channel_data):
            col_means = np.nanmean(channel_data, axis=0)
            if np.isnan(col_means).any():
                return None
            return np.where(np.isnan(channel_data), col_means, channel_data)

        sensors = [
            ("hand_accelerometer", acc_hand),
            ("hand_gyroscope", gyr_hand),
            ("chest_accelerometer", acc_chest),
            ("chest_gyroscope", gyr_chest),
            ("ankle_accelerometer", acc_ankle),
            ("ankle_gyroscope", gyr_ankle),
        ]

        w_features = {}
        skip_window = False
        for mod_name, mod_data in sensors:
            filled = _fill_missing(mod_data)
            if filled is None:
                skip_window = True
                break
            if missing_ratio is not None:
                if np.random.rand() <= missing_ratio:
                    filled = np.zeros_like(filled)
            mod_features = process_by_mod(mod_name, filled)
            if mod_features is None:
                skip_window = True
                break
            w_features.update(mod_features)

        if skip_window:
            start_idx += window_size
            end_idx += window_size
            continue

        data.append(
            dict(
                user_id=user_id,
                label=activity,
                features=w_features,
                sample_id=start_idx,
            )
        )
        start_idx += window_size
        end_idx += window_size
    return data


def run(path=PAMAP2_PATH, out_dir=OUT_DIR, num_workers=96, seed=0, missing_ratio=None):
    np.random.seed(seed)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    info_path = os.path.join(out_dir, "info.json")
    store_info(info_path)
    print(f"Saved info to {info_path}")
    users = glob(os.path.join(path, "*.dat"))
    all_data = []
    with Pool(processes=num_workers) as pool:
        worker = partial(preprocess, missing_ratio=missing_ratio)
        for user_data in pool.imap_unordered(worker, users):
            all_data.extend(user_data)
    print(f"Total number of samples: {len(all_data)}")
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
