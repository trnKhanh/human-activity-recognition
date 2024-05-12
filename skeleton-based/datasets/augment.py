import numpy as np


def moving_augment(
    sample: np.ndarray,
    scale_range=(-0.5, 0.5),
    translate_range=(0, 0),
):
    scale = (
        np.random.rand() * (scale_range[1] - scale_range[0])
        + scale_range[0]
        + 1
    )
    translate = (
        np.random.rand() * (translate_range[1] - translate_range[0])
        + translate_range[0]
    )
    C, T, V, M = sample.shape
    scale_arr = np.linspace(1, scale, T)
    translate_arr = np.linspace(0, translate, T)
    scale_arr = np.tile(scale_arr, (C, 1))
    translate_arr = np.tile(translate_arr, (C, 1))

    aug_sample = (sample.T * scale_arr.T + translate_arr.T).T

    return aug_sample


class UniSampling(object):
    def __init__(self, new_length: int):
        self.new_length = new_length

    def __call__(self, sample: np.ndarray, train=True):
        C, T, V, M = sample.shape

        chunk_size = T / self.new_length
        aug_ids = np.linspace(0, T, self.new_length + 1)[:-1]
        if train:
            choosen = np.random.rand(self.new_length) * (chunk_size - 1)
        else:
            choosen = np.zeros(self.new_length)
        aug_ids = np.round(choosen + aug_ids).astype(np.int32)

        aug_sample = sample[:, aug_ids, :, :]

        return aug_sample