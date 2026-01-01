import os
import json
import datasets


class DataLoader:
    def __init__(self, data_path):
        info_path = os.path.join(data_path, "info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Data path {data_path} is missing info.json.")

        self.metadata = json.load(open(info_path, "r", encoding="utf-8"))
        self.dataset = datasets.load_from_disk(data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]
        return record["test"], record["examples"]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def get_metadata(self):
        return self.metadata

    def get_modalities(self):
        return list(self.metadata["modality_data"].keys())

    def get_task_info(self):
        task_info = f"**Task**:\n{self.metadata['task']}\n\n"
        classes_info = [f"  - {k}: {v}" for k, v in self.metadata["class"].items()]
        classes_info = "\n".join(classes_info)
        task_info += f"**Classes**:\n{classes_info}"
        return task_info

    def get_classes_info(self):
        classes_info = [k for k in self.metadata["class"].keys()]
        return classes_info

    def get_modality_info_dict(self):
        modality_info_dict = {}
        for modality in self.get_modalities():
            data_collection_info = self.metadata["modality_data"][modality]
            feature_extraction_info = self.metadata["feature"][modality.split("_")[-1]]
            modality_info = {}
            modality_info["Data collection"] = data_collection_info.strip()
            modality_info["Feature extraction"] = feature_extraction_info.strip()
            modality_info_dict[modality] = modality_info
        return modality_info_dict
