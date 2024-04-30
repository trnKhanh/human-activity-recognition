import numpy as np
import os
from tqdm import tqdm
from argparse import ArgumentParser
import cv2 as cv


def create_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-dir", required=True, type=str, help="Path to numpy data"
    )
    parser.add_argument(
        "--data-dir-2d", required=True, type=str, help="Path to 2d numpy data"
    )
    parser.add_argument(
        "--extra-data-dir",
        default="",
        type=str,
        help="Path to extra numpy data",
    )
    parser.add_argument(
        "--extra-data-dir-2d",
        default="",
        type=str,
        help="Path to extra 2d numpy data",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Where to save videos"
    )

    return parser.parse_args()


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


def main(args):
    if not os.path.isdir(args.data_dir):
        raise RuntimeError(f"Directory {args.data_dir} not found")
    if len(args.extra_data_dir) > 0 and not os.path.isdir(args.extra_data_dir):
        raise RuntimeError(f"Directory {args.extra_data_dir} not found")
    os.makedirs(args.output_dir, exist_ok=True)

    width, height = 1920, 1080
    exists = dict()
    for file in os.scandir(args.data_dir):
        tmp = file.name.find("A")
        action_id = file.name[tmp + 1 : tmp + 4]
        if action_id in exists:
            continue
        exists[action_id] = 1

        data = np.load(file.path, allow_pickle=True)
        path_2d = os.path.join(args.data_dir_2d, file.name)
        data_2d = np.load(path_2d, allow_pickle=True)
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        video = cv.VideoWriter(
            os.path.join(args.output_dir, f"{action_id}.mp4"),
            fourcc,
            20,
            (width, height * 2),
        )
        M, F, J, C = data.shape
        print(action_id)
        print(data.shape)
        print(data_2d.shape)
        for f in range(F):
            frame = np.zeros((height * 2, width, 3), np.uint8)
            cv.line(frame, (0, height), (width, height), (0, 255, 0))
            for m in range(M):
                for u, v in ADJ:
                    cv.line(
                        frame,
                        (
                            int(500 * data[m, f, u - 1, 0] + 960),
                            int(-500 * data[m, f, u - 1, 1] + 540),
                        ),
                        (
                            int(500 * data[m, f, v - 1, 0] + 960),
                            int(-500 * data[m, f, v - 1, 1] + 540),
                        ),
                        (0, 0, 255),
                        2,
                    )
                    cv.line(
                        frame,
                        (
                            int(data_2d[m, f, u - 1, 0]),
                            int(data_2d[m, f, u - 1, 1] + height),
                        ),
                        (
                            int(data_2d[m, f, v - 1, 0]),
                            int(data_2d[m, f, v - 1, 1] + height),
                        ),
                        (0, 0, 255),
                        2,
                    )
            video.write(frame)
        video.release()
    if len(args.extra_data_dir) > 0:
        for file in os.scandir(args.extra_data_dir):
            tmp = file.name.find("A")
            action_id = file.name[tmp + 1 : tmp + 4]
            if action_id in exists:
                continue
            exists[action_id] = 1

            data = np.load(file.path, allow_pickle=True)
            path_2d = os.path.join(args.extra_data_dir_2d, file.name)
            data_2d = np.load(path_2d, allow_pickle=True)
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            video = cv.VideoWriter(
                os.path.join(args.output_dir, f"{action_id}.mp4"),
                fourcc,
                20,
                (width, height * 2),
            )
            M, F, J, C = data.shape
            print(action_id)
            print(data.shape)
            print(data_2d.shape)
            for f in range(F):
                frame = np.zeros((height * 2, width, 3), np.uint8)
                cv.line(frame, (0, height), (width, height), (0, 255, 0))
                for m in range(M):
                    for u, v in ADJ:
                        cv.line(
                            frame,
                            (
                                int(500 * data[m, f, u - 1, 0] + 960),
                                int(-500 * data[m, f, u - 1, 1] + 540),
                            ),
                            (
                                int(500 * data[m, f, v - 1, 0] + 960),
                                int(-500 * data[m, f, v - 1, 1] + 540),
                            ),
                            (0, 0, 255),
                            2,
                        )
                        if np.isnan(np.sum(data_2d[m, f, u - 1])):
                            continue
                        if np.isnan(np.sum(data_2d[m, f, v - 1])):
                            continue
                        cv.line(
                            frame,
                            (
                                int(data_2d[m, f, u - 1, 0]),
                                int(data_2d[m, f, u - 1, 1] + height),
                            ),
                            (
                                int(data_2d[m, f, v - 1, 0]),
                                int(data_2d[m, f, v - 1, 1] + height),
                            ),
                            (0, 0, 255),
                            2,
                        )
                video.write(frame)
            video.release()


if __name__ == "__main__":
    args = create_args()
    main(args)
