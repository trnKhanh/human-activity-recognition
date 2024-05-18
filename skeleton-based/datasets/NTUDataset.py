import os
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from datasets.augment import ResizeSequence


class NTUDataset(Dataset):
    def __init__(
        self,
        data_path,
        extra_data_path="",
        mode="train",
        split="x-subject",
        length_t=64,
        features="j",
    ):
        super().__init__()
        self.transform = ResizeSequence(length_t)
        self.features = features
        self.samples = {"train": [], "valid": []}
        self.labels = {"train": [], "valid": []}
        self.mode = mode
        if self.mode not in ["train", "valid"]:
            raise NameError(f"Mode {self.mode} is invalid")
        train_ids = []
        if split == "x-subject":
            train_ids = [
                1,
                2,
                4,
                5,
                8,
                9,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                25,
                27,
                28,
                31,
                34,
                35,
                38,
                45,
                46,
                47,
                49,
                50,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                70,
                74,
                78,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                89,
                91,
                92,
                93,
                94,
                95,
                97,
                98,
                100,
                103,
            ]
        elif split == "x-setup":
            train_ids = [i for i in range(2, 33, 2)]
        elif split == "x-view":
            train_ids = [2, 3]
        else:
            raise NameError(f"Split {split} is invalid")

        for file in os.scandir(data_path):
            if split == "x-subject":
                c = "P"
            elif split == "x-setup":
                c = "S"
            else:
                c = "C"
            id = int(file.name[file.name.find(c) + 1 : file.name.find(c) + 4])
            label = (
                int(
                    file.name[file.name.find("A") + 1 : file.name.find("A") + 4]
                )
                - 1
            )
            if id in train_ids:
                self.samples["train"].append(file.path)
                self.labels["train"].append(label)
            else:
                self.samples["valid"].append(file.path)
                self.labels["valid"].append(label)
        if len(extra_data_path) > 0:
            for file in os.scandir(extra_data_path):
                if split == "x-subject":
                    c = "P"
                elif split == "x-setup":
                    c = "S"
                else:
                    c = "C"
                id = int(
                    file.name[file.name.find(c) + 1 : file.name.find(c) + 4]
                )
                label = (
                    int(
                        file.name[
                            file.name.find("A") + 1 : file.name.find("A") + 4
                        ]
                    )
                    - 1
                )
                if id in train_ids:
                    self.samples["train"].append(file.path)
                    self.labels["train"].append(label)
                else:
                    self.samples["valid"].append(file.path)
                    self.labels["valid"].append(label)

    def __len__(self):
        return len(self.samples[self.mode])

    def __getitem__(self, index):
        sample_path = self.samples[self.mode][index]
        label = self.labels[self.mode][index]

        sample = np.load(sample_path, allow_pickle=True)
        sample = torch.from_numpy(sample)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.features[0] == "j":
            data = sample[:3]
        elif self.features[0] == "b":
            data = sample[3:6]
        elif self.features[0] == "m":
            data = sample[6:9]
        else:
            raise ValueError(f"{self.features} is invalid")

        for f in self.features[1:]:
            if f == "j":
                data = torch.cat([data, sample[:3]])
            elif f == "b":
                data = torch.cat([data, sample[3:6]])
            elif f == "m":
                data = torch.cat([data, sample[6:9]])
            else:
                raise ValueError(f"{self.features} is invalid")

        return data, label
