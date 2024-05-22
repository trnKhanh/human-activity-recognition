import os
import numpy as np

import torch
from torch.utils.data import Dataset
from src.datasets.augment import ResizeSequence
from src.datasets.utils import get_angular_motion

from src.graph.ntu_graph import Graph
from src.datasets.utils import get_angular_motion


class NTUDataset(Dataset):
    train_subjects_file = "../../resources/train_subjects.txt"
    train_cameras_file = "../../resources/train_cameras.txt"

    def __init__(
        self,
        data_path,
        extra_data_path="",
        mode="train",
        split="x-subject",
        length_t=64,
        features="j",
        center=20,
    ):
        super().__init__()
        self.graph = Graph(center)

        self.transform = ResizeSequence(length_t)
        self.features = features
        self.samples = {"train": [], "valid": []}
        self.labels = {"train": [], "valid": []}
        self.mode = mode
        if self.mode not in ["train", "valid"]:
            raise NameError(f"Mode {self.mode} is invalid")
        train_ids = []
        if split == "x-subject":
            train_ids = []
            with open(
                os.path.join(
                    os.path.dirname(__file__), self.train_subjects_file
                ),
                "r",
            ) as f:
                lines = f.readlines()
                for line in lines:
                    train_ids.append(int(line))
        elif split == "x-setup":
            train_ids = [i for i in range(2, 33, 2)]
        elif split == "x-view":
            train_ids = []
            with open(
                os.path.join(
                    os.path.dirname(__file__), self.train_cameras_file
                ),
                "r",
            ) as f:
                lines = f.readlines()
                for line in lines:
                    train_ids.append(int(line))
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

        features = self.features.split(",")
        for id, f in enumerate(features):
            features[id] = f.strip()

        data = None

        for f in features:
            if f == "j":
                data = torch.cat([data, sample]) if data is not None else sample
            elif f == "b":
                data = (
                    torch.cat([data, self.__get_bones(sample)])
                    if data is not None
                    else self.__get_bones(sample)
                )
            elif f == "jm":
                data = (
                    torch.cat([data, self.__get_motion(sample)])
                    if data is not None
                    else self.__get_motion(sample)
                )
            elif f == "bm":
                data = (
                    torch.cat([data, self.__get_bones_motion(sample)])
                    if data is not None
                    else self.__get_bones_motion(sample)
                )
            elif f == "am":
                data = (
                    torch.cat([data, self.__get_angular_motion(sample)])
                    if data is not None
                    else self.__get_angular_motion(sample)
                )
            else:
                raise ValueError(f"Feature {f} is invalid")

        if data is not None and self.transform is not None:
            data = self.transform(data)

        return data, label

    def __get_motion(self, sample: torch.Tensor):
        C, T, V, M = sample.size()
        diff = sample[:, 1:, :, :] - sample[:, :-1, :, :]

        joints_motion = torch.zeros_like(sample)
        joints_motion[:, 1:, :, :] = diff

        return joints_motion

    def __get_bones(self, sample: torch.Tensor):
        C, T, V, M = sample.size()
        bones = torch.zeros_like(sample)

        A = self.graph.get_A()
        num_joints = A.shape[0]
        for u in range(num_joints):
            for v in range(num_joints):
                if A[u, v] == 0:
                    continue
                if self.graph.depth[u] > self.graph.depth[v]:
                    bones[:, :, u, :] = sample[:, :, u, :] - sample[:, :, v, :]
        return bones

    def __get_bones_motion(self, sample: torch.Tensor):
        C, T, V, M = sample.size()
        bones = self.__get_bones(sample)

        bones_motion = self.__get_motion(bones)

        return bones_motion

    def __get_angular_motion(self, sample: torch.Tensor):
        return get_angular_motion(sample)
