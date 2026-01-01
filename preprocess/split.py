import os
from collections import defaultdict
from tqdm import tqdm
import shutil
import datasets
import numpy as np
from fire import Fire
from datasets import Dataset


DATA_PATH = "./wesad_output"  # path to the wesad dataset
OUTPUT_PATH = "./wesad_output/splits"  # path to the output directory


def split(
    data_path=DATA_PATH,
    test_per_class=50,
    examples_per_class=1,
    output_path=OUTPUT_PATH,
    seed=0,
    missing=False,
):
    if missing:
        target_path = os.path.join(data_path, "processed_missing")
    else:
        target_path = os.path.join(data_path, "processed")

    reference_path = os.path.join(data_path, "processed")

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target dataset path not found: {target_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference dataset path not found: {reference_path}")

    target_dataset = datasets.load_from_disk(target_path)
    reference_dataset = (
        target_dataset
        if target_path == reference_path
        else datasets.load_from_disk(reference_path)
    )

    classes = target_dataset.unique("label")
    rng = np.random.default_rng(seed)

    target_dataset = target_dataset.shuffle(seed=seed)

    if target_path == reference_path:
        reference_dataset = target_dataset
    else:
        reference_dataset = reference_dataset.shuffle(seed=seed)

    labels_target = target_dataset["label"]
    user_ids_target = target_dataset["user_id"]
    sample_ids_target = (
        target_dataset["sample_id"]
        if "sample_id" in target_dataset.column_names
        else list(range(len(target_dataset)))
    )
    sample_ids_reference = (
        reference_dataset["sample_id"]
        if "sample_id" in reference_dataset.column_names
        else list(range(len(reference_dataset)))
    )

    indices_by_label_target = defaultdict(list)
    indices_by_user_and_label_reference = defaultdict(lambda: defaultdict(list))

    for idx, (label, user_id) in enumerate(zip(labels_target, user_ids_target)):
        indices_by_label_target[label].append(idx)

    for idx, (label, user_id) in enumerate(
        zip(reference_dataset["label"], reference_dataset["user_id"])
    ):
        indices_by_user_and_label_reference[user_id][label].append(idx)

    for c in indices_by_label_target.keys():
        print(f"Label {c} count: {len(indices_by_label_target[c])}")

    split_prefix = "default"
    if missing:
        split_prefix = "missing"

    output_path = os.path.join(
        output_path,
        f"{split_prefix}_ex_{examples_per_class}_seed_{seed}",
    )

    split_data = []

    for c in tqdm(classes, desc="Splitting classes"):
        class_cnt = 0
        class_indices = indices_by_label_target[c]
        class_indices = rng.permutation(class_indices).tolist()

        for idx in tqdm(class_indices, desc="Splitting class data"):
            class_data = target_dataset[idx]
            user_id = class_data["user_id"]
            test_sample_id = sample_ids_target[idx]

            example_indices = []
            valid = True

            for uc in classes:
                candidate_indices = [
                    cand_idx
                    for cand_idx in indices_by_user_and_label_reference[user_id][uc]
                    if sample_ids_reference[cand_idx] != test_sample_id
                ]

                if len(candidate_indices) < examples_per_class:
                    valid = False
                    break

                candidate_indices = candidate_indices.copy()
                rng.shuffle(candidate_indices)
                example_indices.extend(candidate_indices[:examples_per_class])

            if not valid:
                continue

            example_records = [reference_dataset[ex_idx] for ex_idx in example_indices]

            split_data.append(
                dict(
                    test=class_data,
                    examples=example_records,
                )
            )

            class_cnt += 1
            if class_cnt >= test_per_class:
                break

        if class_cnt < test_per_class:
            print(f"Not enough samples for class {c}: {class_cnt}")
            raise ValueError(f"Not enough samples for class {c}: {class_cnt}")

    for record in split_data:
        test_features = record["test"]["features"]
        test_sample_id = record["test"]["sample_id"]
        test_user_id = record["test"]["user_id"]
        for example in record["examples"]:
            if example["user_id"] != test_user_id:
                raise ValueError(
                    f"Example user {example['user_id']} differs from test user {test_user_id}"
                )
            if (
                example["sample_id"] == test_sample_id
                and example["features"] == test_features
            ):
                raise ValueError(
                    f"Example sample shares identical features with test sample {test_sample_id}"
                )

    split_dataset = Dataset.from_list(split_data)
    split_dataset.save_to_disk(output_path)
    print(f"Split dataset saved to {output_path}")

    info_path = os.path.join(data_path, "info.json")
    shutil.copy(info_path, os.path.join(output_path, "info.json"))


if __name__ == "__main__":
    Fire(split)
