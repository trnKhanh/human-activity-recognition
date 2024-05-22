import numpy as np


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


def convert_to_numpy(data):
    num_frames = data["num_frames"]

    bodies_data = dict()
    bodies_data["num_frames"] = num_frames
    bodies_data["data"] = dict()

    for f_id in range(num_frames):
        frame = data["frames"][f_id]

        num_bodies = frame["num_bodies"]
        for b_id in range(num_bodies):
            body = frame["bodies"][b_id]
            body_id = body["body_id"]
            if body_id not in bodies_data["data"]:
                bodies_data["data"][body_id] = {
                    "start": f_id,
                    "data_3d": [],
                    "data_2d": [],
                }

            num_joints = body["num_joints"]
            joints_3d_np = np.zeros((num_joints, 3))
            joints_2d_np = np.zeros((num_joints, 2))
            for j_id in range(num_joints):
                joint = body["joints"][j_id]

                joints_3d_np[j_id, 0] = joint["x"]
                joints_3d_np[j_id, 1] = joint["y"]
                joints_3d_np[j_id, 2] = joint["z"]

                joints_2d_np[j_id, 0] = joint["color_x"]
                joints_2d_np[j_id, 1] = joint["color_y"]

            bodies_data["data"][body_id]["data_3d"].append(joints_3d_np)
            bodies_data["data"][body_id]["data_2d"].append(joints_2d_np)

    for body_id in bodies_data["data"].keys():
        bodies_data["data"][body_id]["data_3d"] = np.stack(
            bodies_data["data"][body_id]["data_3d"], axis=0
        )
        bodies_data["data"][body_id]["data_2d"] = np.stack(
            bodies_data["data"][body_id]["data_2d"], axis=0
        )
        bodies_data["data"][body_id]["motion"] = np.sum(
            np.var(
                bodies_data["data"][body_id]["data_3d"].reshape(-1, 3), axis=0
            )
        )

    return bodies_data
