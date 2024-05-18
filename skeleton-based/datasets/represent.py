import numpy as np
import os
from tqdm import tqdm
from argparse import ArgumentParser
from augment import moving_augment, UniSampling
import cv2 as cv
from NTUDataset import NTUDataset


def create_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-dir", required=True, type=str, help="Path to numpy data"
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
    parser.add_argument(
        "--classes-path", required=True, type=str, help="Path to classes file"
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


def load_classes(args):
    classes = dict()
    with open(args.classes_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            id, name = line.split(".")[:2]
            classes[int(id)] = name.strip().replace(" ", "-").replace("/", "|")
    return classes


def main(args):
    if not os.path.isdir(args.data_dir):
        raise RuntimeError(f"Directory {args.data_dir} not found")
    if len(args.extra_data_dir) > 0 and not os.path.isdir(args.extra_data_dir):
        raise RuntimeError(f"Directory {args.extra_data_dir} not found")
    os.makedirs(args.output_dir, exist_ok=True)
    classes = load_classes(args)

    width, height = 1920, 1080
    fps = 24
    exists = dict()
    dataset = NTUDataset(
        data_path=args.data_dir,
        extra_data_path=args.extra_data_dir,
        mode="train",
        split="x-subject",
        features="jbm",
        length_t=64,
    )
    for i in range(len(dataset)):
        data, action_id = dataset[i]
        action_id += 1
        if action_id in exists:
            continue
        exists[action_id] = 1

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        video = cv.VideoWriter(
            os.path.join(
                args.output_dir, f"{action_id}_{classes[action_id]}.mp4"
            ),
            fourcc,
            fps,
            (width, height * 2),
        )

        C, T, V, M = data.shape

        print(action_id)
        print(data.shape)
        for f in range(T):
            frame = np.zeros((height * 2, width, 3), np.uint8)
            cv.line(frame, (0, height), (width, height), (0, 255, 0))
            for m in range(M):
                for u, v in ADJ:
                    cv.line(
                        frame,
                        (
                            int(500 * data[0, f, u - 1, m] + 960),
                            int(-500 * data[1, f, u - 1, m] + 540),
                        ),
                        (
                            int(500 * data[0, f, v - 1, m] + 960),
                            int(-500 * data[1, f, v - 1, m] + 540),
                        ),
                        (0, 0, 255),
                        2,
                    )
                for u in range(25):
                    x = u // 5
                    y = u % 5
                    cv.circle(
                        frame,
                        (x * 192 + 96, y * 192 + 50 + height),
                        10,
                        (0, 0, 255),
                    )
                    cv.line(
                        frame,
                        (
                            int(x * 192 + 96),
                            int(y * 192 + 50) + height,
                        ),
                        (
                            int(x * 192 + 500 * data[3, f, u, m] + 96),
                            int(y * 192 + -500 * data[4, f, u, m] + 50)
                            + height,
                        ),
                        (0, 0, 255),
                        2,
                    )
                    cv.circle(
                        frame,
                        (x * 192 + 96 + width // 2, y * 192 + 50 + height),
                        10,
                        (0, 255, 255),
                    )
                    cv.arrowedLine(
                        frame,
                        (
                            int(x * 192 + 96) + width // 2,
                            int(y * 192 + 50) + height,
                        ),
                        (
                            int(x * 192 + 500 * data[6, f, u, m] + 96)
                            + width // 2,
                            int(y * 192 + -500 * data[7, f, u, m] + 50)
                            + height,
                        ),
                        (0, 255, 255),
                        2,
                    )
            video.write(frame)
        video.release()


if __name__ == "__main__":
    args = create_args()
    main(args)
