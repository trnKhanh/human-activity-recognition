import numpy as np
import os
import copy as cp
from tqdm import tqdm
from argparse import ArgumentParser


def create_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-dir", required=True, type=str, help="Path to text data"
    )
    parser.add_argument(
        "--save-dir", required=True, type=str, help="Where to save numpy data"
    )
    parser.add_argument(
        "--missing-files", required=True, type=str, help="Path to missing files"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existed file",
    )
    parser.add_argument(
        "--max-frames",
        default=300,
        type=int,
        help="Max frames (default: 300)",
    )
    parser.add_argument(
        "--max-bodies",
        default=2,
        type=int,
        help="Max bodies (default: 2)",
    )
    parser.add_argument(
        "--num-joints",
        default=25,
        type=int,
        help="Number of joints (default: 25)",
    )

    return parser.parse_args()


def main(args):
    if not os.path.isdir(args.data_dir):
        raise RuntimeError(f"Directory {args.data_dir} not found")
    if not os.path.isfile(args.missing_files):
        raise RuntimeError(f"File {args.missing_files} not found")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + "_2d", exist_ok=True)

    missing_files = load_missing_files(args)
    for file in tqdm(os.scandir(args.data_dir)):
        if file.name.split(".")[-1] != "skeleton":
            continue

        if file.name not in missing_files:
            save_name = file.name.split(".")[0] + ".npy"
            save_path = os.path.join(args.save_dir, save_name)

            if not args.overwrite and os.path.isfile(save_path):
                continue

            data = load_skeleton_data(file.path)

            np_data3d, np_data2d = convert_to_numpy(
                data, args.max_frames, args.max_bodies, args.num_joints
            )
            np.save(save_path, np_data3d)

            save_2d_path = os.path.join(args.save_dir + "_2d", save_name)
            np.save(save_2d_path, np_data2d)


def dfs(u, edges, par):
    for e in edges:
        if u == e[0] - 1:
            v = e[1] - 1
        elif u == e[1] - 1:
            v = e[0] - 1
        else:
            continue
        if par[v] != -1:
            continue

        par[v] = u
        dfs(v, edges, par)


ADJ = [
    (1, 2),
    (2, 21),
    (3, 21),
    (4, 3),
    (5, 21),
    (6, 5),
    (7, 6),
    (8, 7),
    (9, 21),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 1),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 1),
    (18, 17),
    (19, 18),
    (20, 19),
    (22, 23),
    (23, 8),
    (24, 25),
    (25, 12),
]


def convert_to_numpy(data, max_frame=300, num_bodies=2, num_joints=25):
    par = [-1 for _ in range(num_joints)]
    par[20] = -2
    dfs(20, ADJ, par)
    np_data3d = np.zeros(
        (num_bodies, 300, num_joints, 3 + 3 + 3), dtype=np.float32
    )
    np_data2d = np.zeros((num_bodies, 300, num_joints, 2), dtype=np.float32)

    bodies = dict()

    for fidx, frame in enumerate(data["frames"]):
        for body in frame["bodies"]:
            body_id = body["body_id"]

            if body_id not in bodies:
                bodies[body_id] = dict(kps3d=[], kps2d=[], start=fidx)

            assert body["num_joints"] == num_joints

            np_joints3d = np.zeros((num_joints, 3))
            np_joints2d = np.zeros((num_joints, 2))

            for jidx, joint in enumerate(body["joints"]):
                np_joints3d[jidx, :] = joint["x"], joint["y"], joint["z"]
                np_joints2d[jidx, :] = joint["color_x"], joint["color_y"]

            bodies[body_id]["kps3d"].append(np_joints3d)
            bodies[body_id]["kps2d"].append(np_joints2d)

    bodies = bodies.values()

    for body in bodies:
        body["kps3d"] = np.stack(body["kps3d"])
        body["kps2d"] = np.stack(body["kps2d"])

    bodies = denoise(bodies)

    for i in range(len(bodies)):
        s = bodies[i]["start"]
        e = bodies[i]["start"] + bodies[i]["kps3d"].shape[0]
        assert bodies[i]["kps3d"].shape[0] == bodies[i]["kps2d"].shape[0]
        bones = np.zeros((e - s, num_joints, 3))
        motion = np.zeros((e - s, num_joints, 3))
        assert bones.shape == motion.shape
        assert bones.shape == bodies[i]["kps3d"].shape
        for f in range(e - s):
            for j in range(num_joints):
                if par[j] != -2:
                    bones[f, j, :] = (
                        bodies[i]["kps3d"][f, j, :]
                        - bodies[i]["kps3d"][f, par[j], :]
                    )
                if f != 0:
                    motion[f, j, :] = (
                        bodies[i]["kps3d"][f, j, :]
                        - bodies[i]["kps3d"][f - 1, j, :]
                    )
        np_data3d[i, s:e, :, :3] = bodies[i]["kps3d"]
        np_data3d[i, s:e, :, 3:6] = bones
        np_data3d[i, s:e, :, 6:9] = motion
        np_data2d[i, s:e, :, :2] = bodies[i]["kps2d"]

    np_data3d = compress(np_data3d, max_frame)
    np_data2d = compress(np_data2d, max_frame)

    np_data3d = np_data3d.transpose(3, 1, 2, 0)
    np_data2d = np_data2d.transpose(3, 1, 2, 0)

    return np_data3d, np_data2d


def compress(data: np.ndarray, new_length: int):
    M, T, V, C = data.shape
    if T == new_length:
        return data

    rate = T / new_length
    new_data = np.zeros((M, new_length, V, C), dtype=np.float32)

    i = 0
    cur_frame = 0
    if rate >= 1:
        while cur_frame <= T and i < new_length:
            new_data[:, i, :, :] = data[:, int(np.round(cur_frame)), :, :]
            i += 1
            cur_frame += rate
    else:
        new_data[:, :T, :, :] = data

    return new_data


def denoise(bodies):
    for body in bodies:
        body["valid"] = True

    bodies = [body for body in bodies if body["valid"]]
    if len(bodies) == 0:
        raise RuntimeError("Body len is 0")

    for body in bodies:
        body["motion"] = np.sum(np.std(np.vstack(body["kps3d"]), axis=0))

    if len(bodies) == 1:
        return bodies

    bodies.sort(key=lambda x: -x["motion"])

    return bodies[:2]


def load_skeleton_data(file_path: str):
    data = dict()
    file_name = file_path.split("/")[-1]
    data["setup_number"] = int(file_name[1:4])
    data["camera_id"] = int(file_name[5:8])
    data["subject_id"] = int(file_name[9:12])
    data["replication_number"] = int(file_name[13:16])
    data["label"] = int(file_name[17:20])
    with open(file_path, "r") as f:
        data["num_frames"] = int(f.readline())
        data["frames"] = []
        for _ in range(data["num_frames"]):
            frame_info = dict()
            frame_info["num_bodies"] = int(f.readline())
            frame_info["bodies"] = []

            for _ in range(frame_info["num_bodies"]):
                body_datas = f.readline().split(" ")
                body_info = dict()
                body_info["body_id"] = int(body_datas[0])
                body_info["clipped_edges"] = int(body_datas[1])
                body_info["hand_left_conf"] = int(body_datas[2])
                body_info["hand_left_state"] = int(body_datas[3])
                body_info["hand_right_conf"] = int(body_datas[4])
                body_info["hand_right_state"] = int(body_datas[5])
                body_info["is_restricted"] = int(body_datas[6])
                body_info["lean_x"] = float(body_datas[7])
                body_info["lean_y"] = float(body_datas[8])
                body_info["tracking_state"] = int(body_datas[9])

                body_info["num_joints"] = int(f.readline())
                body_info["joints"] = []
                for _ in range(body_info["num_joints"]):
                    joint_datas = f.readline().split(" ")
                    joint_info = dict()

                    # 3D location of the joint
                    joint_info["x"] = float(joint_datas[0])
                    joint_info["y"] = float(joint_datas[1])
                    joint_info["z"] = float(joint_datas[2])

                    # 2D location of the joint in corresponding depth/IR frame
                    joint_info["depth_x"] = float(joint_datas[3])
                    joint_info["depth_y"] = float(joint_datas[4])

                    # 2D location of the joint in corresponding RGB frame
                    joint_info["color_x"] = float(joint_datas[5])
                    joint_info["color_y"] = float(joint_datas[6])

                    # Orientations of the joint
                    joint_info["orientation_w"] = float(joint_datas[7])
                    joint_info["orientation_x"] = float(joint_datas[8])
                    joint_info["orientation_y"] = float(joint_datas[9])
                    joint_info["orientation_z"] = float(joint_datas[10])

                    # Tracking state of the joint
                    joint_info["tracking_state"] = int(joint_datas[11])

                    body_info["joints"].append(joint_info)

                assert body_info["num_joints"] == len(body_info["joints"])
                assert body_info["num_joints"] == 25
                frame_info["bodies"].append(body_info)

            assert frame_info["num_bodies"] == len(frame_info["bodies"])
            data["frames"].append(frame_info)
        assert data["num_frames"] == len(data["frames"])

    return data


def load_missing_files(args):
    missing_files = dict()
    with open(args.missing_files, "r") as f:
        lines = f.readlines()
        for line in lines:
            missing_files[line[:-1] + ".skeleton"] = True

    return missing_files


if __name__ == "__main__":
    args = create_args()
    main(args)
