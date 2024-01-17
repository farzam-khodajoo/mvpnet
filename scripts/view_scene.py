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

from mvpnet.utils.o3d_util import draw_point_cloud
from mvpnet.utils.visualize import label2color, visualize_labels
from mvpnet.data.scannet_2d3d import ScanNet2D3DChunksTest
from torch.utils.data.dataloader import DataLoader
from utils.general import (
    raise_path_error,
    Pickle
)
from utils.o3d import (
    PLYFormat
)
from utils.inference import (
    load_class_mapping,
    get_available_labels
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        required=False,
        help="the target scene id",
    )

    args = parser.parse_args()

    scene = (
            Path(SCANNET_DIRECTORY)
            / args.scene
            / "{}_vh_clean_2.ply".format(args.scene)
    )

    scene_ply = PLYFormat.read_pc_from_ply(scene, return_color=True)
    scene_colors = scene_ply["colors"] / 255.
    scene_point_cloud = draw_point_cloud(scene_ply["points"], colors=scene_colors)

    o3d.visualization.draw_geometries([scene_point_cloud], height=500, width=500, window_name="View from {}".format(args.scene))


if __name__ == "__main__":
    main()