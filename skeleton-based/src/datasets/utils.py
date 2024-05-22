import torch


def get_neighbor_angle_motion(sample: torch.Tensor):
    p = (sample[3:6, 1:, :, :] * sample[3:6, :-1, :, :]).sum(0) + 1e-6
    len = ((sample[3:6, :, :, :] ** 2).sum(0) + 1e-6) ** 0.5

    m_cos = p / (len[1:, :, :] * len[:-1, :, :])
    neighbor_theta = torch.acos(m_cos)

    return 1 - m_cos


def get_global_angle_motion(sample: torch.Tensor, center: int):
    offset = sample[:3, :, :, :] - sample[:3, :, center : center + 1, :]
    p = (offset[:, 1:, :, :] * offset[:, :-1, :, :]).sum(0) + 1e-6
    len = ((offset[:, :, :, :] ** 2).sum(0) + 1e-6) ** 0.5

    m_cos = p / (len[1:, :, :] * len[:-1, :, :])
    global_theta = torch.acos(m_cos)

    return 1 - m_cos


def get_local_angle_motion(sample: torch.Tensor):
    C, T, V, M = sample.size()

    local_center = {
        10: [9, 11, 12, 24, 25],
        6: [5, 7, 8, 22, 23],
        21: [1, 2, 3, 4],
        18: [17, 19, 20],
        14: [13, 15, 16],
    }
    offset = torch.zeros((3, T, V, M))
    for center, v in local_center.items():
        center -= 1
        for id, x in enumerate(v):
            v[id] -= 1

        offset[:, :, v, :] = (
            sample[:3, :, v, :] - sample[:3, :, center : center + 1, :]
        )

    p = (offset[:, 1:, :, :] * offset[:, :-1, :, :]).sum(0) + 1e-6
    len = ((offset[:, :, :, :] ** 2).sum(0) + 1e-6) ** 0.5

    m_cos = p / (len[1:, :, :] * len[:-1, :, :])
    local_theta = torch.acos(m_cos)

    return 1 - m_cos


def get_angular_motion(sample: torch.Tensor):
    C, T, V, M = sample.size()
    angle_embed = torch.zeros((3, T, V, M))

    angle_embed[0, 1:, :, :] = get_neighbor_angle_motion(sample)
    angle_embed[1, 1:, :, :] = get_global_angle_motion(sample, 20)
    angle_embed[2, 1:, :, :] = get_local_angle_motion(sample)

    return angle_embed
