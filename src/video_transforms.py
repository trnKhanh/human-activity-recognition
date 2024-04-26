import torch
import torchvision.transforms as transforms


class VideoResize(object):
    def __init__(
        self, size, interpolation=transforms.InterpolationMode.BILINEAR
    ):
        self.worker = transforms.Resize(size, interpolation)

    def __call__(self, images):
        return [self.worker(image) for image in images]


class VideoCenterCrop(object):
    def __init__(self, size):
        self.worker = transforms.CenterCrop(size)

    def __call__(self, images):
        return [self.worker(image) for image in images]


class VideoNormalize(object):
    def __init__(self, mean, std):
        self.worker = transforms.Normalize(mean, std)

    def __call__(self, images):
        return [self.worker(image) for image in images]


class VideoToTensor(object):
    def __init__(self):
        self.worker = transforms.ToTensor()

    def __call__(self, images):
        return [self.worker(image) for image in images]

