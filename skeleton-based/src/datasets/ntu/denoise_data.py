import os
import numpy as np

NOISE_LENGTH_THRESHOLD = 11
NOISE_SPR_THRESHOLD_1 = 0.8
NOISE_SPR_THRESHOLD_2 = 0.69754


def denoise_by_length(bodies_data):
    new_bodies_data = bodies_data["data"].copy()
    for body_id, body_data in new_bodies_data.items():
        start = body_data["start"]
        length = body_data["data_3d"].shape[0]

        if length <= NOISE_LENGTH_THRESHOLD:
            del bodies_data["data"][body_id]

    return bodies_data


def get_valid_frames(joints):
    num_frames = joints.shape[0]

    valid_frames = []
    for f_id in range(num_frames):
        x = joints[f_id, :, 0]
        y = joints[f_id, :, 1]

        if (x.max() - x.min()) <= NOISE_SPR_THRESHOLD_1 * (y.max() - y.min()):
            valid_frames.append(f_id)

    return valid_frames


def denoise_by_spread(bodies_data):
    new_bodies_data = bodies_data["data"].copy()
    for body_id, body_data in new_bodies_data.items():
        motion = body_data["motion"]
        valid_frames = get_valid_frames(body_data["data_3d"])

        num_frames = body_data["data_3d"].shape[0]
        num_noise = num_frames - len(valid_frames)

        noise_ratio = num_noise / num_frames

        if noise_ratio >= NOISE_SPR_THRESHOLD_2:
            del bodies_data["data"][body_id]
        else:
            valid_joints = bodies_data["data"][body_id]["data_3d"][valid_frames]
            bodies_data["data"][body_id]["motion"] = min(
                motion, np.sum(np.var(valid_joints.reshape((-1, 3)), axis=0))
            )

    return bodies_data


def denoise(bodies_data):
    bodies_data = denoise_by_length(bodies_data)

    if len(bodies_data["data"]) == 1:
        return bodies_data

    bodies_data = denoise_by_spread(bodies_data)
    if len(bodies_data["data"]) == 1:
        return bodies_data

    bodies_data["data"] = dict(
        sorted(
            bodies_data["data"].items(),
            key=lambda x: x[1]["motion"],
            reverse=True,
        )
    )

    return bodies_data


def get_one_actor_data(bodies_data):
    bodies_data_list = list(bodies_data["data"].values())
    T, N, C = bodies_data_list[0]["data_3d"].shape
    sample = bodies_data_list[0]["data_3d"].reshape((1, T, N, C))
    return sample


def get_two_actors_data(bodies_data):
    num_frames = bodies_data["num_frames"]
    bodies_data = denoise(bodies_data)
    bodies_data_list = list(bodies_data["data"].values())

    if len(bodies_data_list) == 1:
        return get_one_actor_data(bodies_data)

    num_joints = bodies_data_list[0]["data_3d"].shape[1]
    assert num_joints == 25

    sample = np.zeros((2, num_frames, num_joints, 3))
    start = [0, 0]
    end = [0, 0]
    for i in range(2):
        num_frames = bodies_data_list[i]["data_3d"].shape[0]

        start[i] = bodies_data_list[i]["start"]
        end[i] = bodies_data_list[i]["start"] + num_frames
        sample[i, start[i] : end[i], :, :] = bodies_data_list[i]["data_3d"]

    i = 2
    while i < len(bodies_data_list):
        num_frames = bodies_data_list[i]["data_3d"].shape[0]

        s = bodies_data_list[i]["start"]
        e = bodies_data_list[i]["start"] + num_frames
        for j in range(2):
            if min(e, end[j]) <= max(s, start[j]):
                sample[j, s:e, :, :] = bodies_data_list[i]["data_3d"]

                start[j] = min(s, start[j])
                end[j] = max(e, end[j])
                break

        i += 1

    return sample


def align_sample(sample, num_frames, num_bodies, num_joints):
    new_sample = np.zeros(
        (num_bodies, num_frames, num_joints, 3), dtype=np.float32
    )
    M, T, N, C = sample.shape
    new_sample[:M, :T, :N, :C] = sample
    return new_sample


def remove_missing_frames(sample):
    valid_frames = np.where(np.sum(sample, axis=(0, 2, 3)) != 0)[0]
    missing_frames = np.where(np.sum(sample, axis=(0, 2, 3)) == 0)[0]

    if len(missing_frames) > 0:
        sample = sample[:, valid_frames, :, :]

    return sample


def get_denoise_data(
    bodies_data,
    num_frames: int = 300,
    num_bodies: int = 2,
    num_joints: int = 25,
):
    if len(bodies_data["data"]) == 1:
        sample = get_one_actor_data(bodies_data)

    else:
        length = len(bodies_data["data"])
        sample = get_two_actors_data(bodies_data)
        shape1 = sample.shape
        sample = remove_missing_frames(sample)
        shape2 = sample.shape
        if length != sample.shape[0]:
            with open(
                os.path.join(
                    os.path.dirname(__file__), "statistics/denoise.txt"
                ),
                "a",
            ) as f:
                f.write(f"{bodies_data['name']}\n")

        # DEBUG
        if shape1 != shape2:
            with open(
                os.path.join(
                    os.path.dirname(__file__), "statistics/missing_frames.txt"
                ),
                "a",
            ) as f:
                f.write(f"{bodies_data['name']}\n")

    sample = align_sample(
        sample,
        num_frames=num_frames,
        num_bodies=num_bodies,
        num_joints=num_joints,
    )
    M, T, V, C = sample.shape
    sample = sample.transpose(3, 1, 2, 0)
    assert sample.shape == (3, num_frames, num_joints, num_bodies)

    return sample
