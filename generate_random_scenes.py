from glob import glob
from pathlib import Path
from random import randint, choice
import shutil
from tqdm import tqdm
import argparse

from SETTING import ROOT_DIRECTORY, SCANNET_DIRECTORY

import logging

logging.basicConfig(level=logging.INFO)


def check_path(title, path):
    if not Path(path).exists():
        logging.error("{} {} does not exists !".format(title, path))
        exit()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="sample which should be searched on dataset",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="sample which should be searched on dataset",
    )

    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="sample which should be searched on dataset",
    )

    args = parser.parse_args()

    check_path("2d_scannet directory", args.dataset)
    check_path("save directory", args.save_dir)

    scannet_2d = glob(str(Path(args.dataset) / "*"))
    logging.info("Available samples: {}".format([Path(s).name for s in scannet_2d]))

    for _ in range(args.count):
        sample_2d = choice(scannet_2d)
        sample_2d = Path(sample_2d)
        sample_color = glob(str(sample_2d / "color" / "*.jpg"))
        ids = [Path(color).name.split(".jpg")[0] for color in sample_color]
        ids = [int(id_) for id_ in ids]
        max_id = max(ids)
        min_id = min(ids)
        random_id = randint(min_id, max_id)

        destination = Path(args.save_dir) / str(random_id)
        Path.mkdir(destination, exist_ok=True)

        color = Path(sample_2d / "color" / "{}.jpg".format(random_id))
        depth = Path(sample_2d / "depth" / "{}.png".format(random_id))
        pose = Path(sample_2d / "pose" / "{}.txt".format(random_id))
        intrinsic = Path(sample_2d / "intrinsic" / "intrinsic_depth.txt")

        shutil.copy(str(color), str(destination))
        shutil.copy(str(depth), str(destination))
        shutil.copy(str(pose), str(destination))
        shutil.copy(str(intrinsic), str(destination))

        with open(Path(destination) / "IGNORE.txt", "w") as ignore_file:
            ignore_file.writelines(str(sample_2d.name))

    print("done generating {} samples into {}".format(args.count, args.save_dir))

if __name__ == "__main__":
    main()
