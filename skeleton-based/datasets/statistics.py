import numpy as np
import os
import copy as cp
from tqdm import tqdm
from argparse import ArgumentParser
from preprocess import load_skeleton_data, load_missing_files


def create_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-dir", required=True, type=str, help="Path to text data"
    )
    parser.add_argument(
        "--extra-data-dir", default="", type=str, help="Path to extra data"
    )
    parser.add_argument(
        "--missing-files", required=True, type=str, help="Path to missing files"
    )
    return parser.parse_args()


def main(args):
    cnt = dict()
    missing_files = load_missing_files(args)
    for file in tqdm(os.scandir(args.data_dir)):
        if file.name.split(".")[-1] != "skeleton":
            continue

        if file.name not in missing_files:
            data = load_skeleton_data(file.path)
            if data["num_frames"] not in cnt:
                cnt[data["num_frames"]] = [0, file.path]
            cnt[data["num_frames"]][0] += 1
    for file in tqdm(os.scandir(args.extra_data_dir)):
        if file.name.split(".")[-1] != "skeleton":
            continue

        if file.name not in missing_files:
            data = load_skeleton_data(file.path)
            if data["num_frames"] not in cnt:
                cnt[data["num_frames"]] = [0, file.path]
            cnt[data["num_frames"]][0] += 1

    cnt = dict(sorted(cnt.items()))
    for k, v in cnt.items():
        print(f"{k}: {v}")
    print(os.get_terminal_size().columns * "=")

    cnt_per50 = dict()
    for k, v in cnt.items():
        if k // 50 not in cnt_per50:
            cnt_per50[k // 50] = 0
        cnt_per50[k // 50] += v[0]
    cnt_per50 = dict(sorted(cnt_per50.items()))
    for k, v in cnt_per50.items():
        print(f"{k * 50}: {v}")


if __name__ == "__main__":
    args = create_args()
    main(args)
