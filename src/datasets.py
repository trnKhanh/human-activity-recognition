import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import json
import cv2 as cv
from PIL import Image

import video_transforms as video_transforms


class VideoDataset(Dataset):
    def __init__(
        self,
        anno_path,
        data_path,
        crop_size,
        img_size,
        transform,
        num_frames,
        sampling_rate,
        mode="train",
        aug=None,
        use_decord=False,
    ):
        self.anno_path = anno_path
        self.data_path = data_path
        self.img_size = img_size
        self.crop_size = crop_size
        self.transform = transform
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.mode = mode
        self.aug = aug
        self.use_decord = use_decord

        self.video_paths, self.labels = self._make_dataset(
            anno_path, data_path, mode
        )

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.video_transform = transforms.Compose(
            [
                video_transforms.VideoResize(self.img_size),
                video_transforms.VideoCenterCrop(self.crop_size),
                video_transforms.VideoToTensor(),
                video_transforms.VideoNormalize(imagenet_mean, imagenet_std),
            ]
        )

    def _make_dataset(self, anno_path, data_path, mode):
        class_id = dict()
        class_id_path = anno_path + "/labels.json"
        with open(class_id_path, "r") as f:
            data = json.load(f)
            for class_name in data.keys():
                class_id[class_name] = int(data[class_name])

        video_paths = []
        labels = []
        if mode == "train":
            label_file = anno_path + "/train.json"
            with open(label_file, "r") as f:
                data = json.load(f)
                for sample in data:
                    video_id = sample["id"]
                    video_label = sample["label"]
                    video_template = (
                        sample["template"].replace("[", "").replace("]", "")
                    )
                    video_path = f"{data_path}/{video_id}.webm"

                    video_paths.append(video_path)
                    labels.append(class_id[video_template])
        elif mode == "validation":
            label_file = anno_path + "/validation.json"
            with open(label_file, "r") as f:
                data = json.load(f)
                for sample in data:
                    video_id = sample["id"]
                    video_label = sample["label"]
                    video_template = (
                        sample["template"].replace("[", "").replace("]", "")
                    )
                    video_path = f"{data_path}/{video_id}.webm"

                    video_paths.append(video_path)
                    labels.append(class_id[video_template])
        elif mode == "test":
            pass
        else:
            raise NameError(f"mode {mode} is invalid")

        return video_paths, labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.labels[index]

        frames = self._read_video(video_path)

        if self.transform is not None:
            frames = self.transform(frames)

        frames = self.video_transform(frames)

        if self.aug is not None:
            frames = self.aug(frames)

        frames = torch.stack(frames)

        frames = frames.transpose(0, 1)

        return frames, label

    def _read_video(self, video_path):
        frames = []
        if not self.use_decord:
            cap = cv.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        else:
            raise NotImplemented()

        duration = len(frames)
        frame_ids = [
            i
            for i in range(
                0,
                min(duration, self.num_frames * self.sampling_rate),
                self.sampling_rate,
            )
        ]

        # Replicate if clip does not have enough frames
        if self.num_frames > len(frame_ids):
            frame_ids.extend(
                [frame_ids[-1] for _ in range(len(frame_ids), self.num_frames)]
            )
        samples = [
            Image.fromarray(frames[id]).convert("RGB") for id in frame_ids
        ]

        return samples
