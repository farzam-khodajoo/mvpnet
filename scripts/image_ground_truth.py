import sys
import os

is_windows = sys.platform == "win32"

extend_path = os.getcwd()
if is_windows:
    extend_path = extend_path.replace("/", "\\")

sys.path.append(extend_path)

from SETTING import ROOT_DIRECTORY, SCANNET_DIRECTORY

from collections import OrderedDict
import os.path as osp
import csv
import logging
import argparse
from pathlib import Path
from glob import glob
import open3d as o3d
import numpy as np
from pathlib import Path
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm
from prettytable import PrettyTable, ALL

import cv2

from mvpnet.data.scannet_2d import ScanNet2D
from mvpnet.utils.o3d_util import draw_point_cloud
from mvpnet.utils.visualize import label2color, visualize_labels
from utils.general import (
    raise_path_error,
    Pickle
)
from utils.cv import show

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="label file to be viewed",
    )

    args = parser.parse_args()

    pickle_directory = Path(ROOT_DIRECTORY)/"pickles"
    scan_directory = Path(ROOT_DIRECTORY)/"scans_resize_160x120"

    if not pickle_directory.exists():
        raise_path_error("Pickle directory", pickle_directory)

    if not scan_directory.exists():
        raise_path_error("Scan directory", pickle_directory)

    from mvpnet.config.sem_seg_2d import cfg
    test_dataset = ScanNet2D(cfg.DATASET.ROOT_DIR, split="test",
                             subsample=None, to_tensor=True,
                             resize=cfg.DATASET.ScanNet2D.resize,
                             normalizer=cfg.DATASET.ScanNet2D.normalizer,
                             )

    if args.label:
        image = cv2.imread(args.label, cv2.IMREAD_UNCHANGED)
        image = test_dataset.label_mapping[image].copy()
        image = label2color(image.flatten())
        image = image.reshape((120, 160, 3))
        show(image, "Ground truth (labels)")


    

if __name__ == "__main__":
    main()