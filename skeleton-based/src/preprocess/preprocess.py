import os
import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser

from src.preprocess.raw_data import load_skeleton_data, convert_to_numpy
from src.preprocess.denoise_data import get_denoise_data


def load_missing_files(args):
    missing_files = dict()
    with open(args.missing_files, "r") as f:
        lines = f.readlines()
        for line in lines:
            missing_files[line[:-1] + ".skeleton"] = True

    return missing_files


def create_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        help="Directory contains skeleton data",
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

    missing_files = load_missing_files(args)
    cnt = 0

    with tqdm(os.scandir(args.data_dir)) as t_log:
        for file in t_log:
            t_log.set_postfix({"file_name": file.name})
            if file.name.split(".")[-1] != "skeleton":
                continue

            if file.name not in missing_files:
                save_name = file.name.split(".")[0] + ".npy"
                save_path = os.path.join(args.save_dir, save_name)

                if not args.overwrite and os.path.isfile(save_path):
                    continue

                data = load_skeleton_data(file.path)
                bodies_data = convert_to_numpy(data)
                bodies_data["name"] = file.name

                sample = get_denoise_data(bodies_data)
                assert sample.shape == (3, 300, 25, 2)
                np.save(save_path, sample)


if __name__ == "__main__":
    args = create_args()
    main(args)
