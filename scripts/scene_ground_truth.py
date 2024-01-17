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
    parser.add_argument(
        "--save",
        type=str,
        required=False,
        help="save segmentation result into pickle",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        default=False,
        help="get list of all available scene ids",
    )

    args = parser.parse_args()

    pickle_directory = Path(ROOT_DIRECTORY)/"pickles"
    scan_directory = Path(ROOT_DIRECTORY)/"scans_resize_160x120"

    if not pickle_directory.exists():
        raise_path_error("Pickle directory", pickle_directory)

    if not scan_directory.exists():
        raise_path_error("Scan directory", pickle_directory)

    k = 3
    num_views = 5
    min_nb_pts = 2048
    test_dataset = ScanNet2D3DChunksTest(
        cache_dir=str(pickle_directory),
        image_dir=str(scan_directory),
        split="test",
        chunk_size=[1.5, 1.5],
        chunk_stride=0.5,
        chunk_thresh=1000,
        num_rgbd_frames=num_views,
        resize=(160, 120),
        image_normalizer=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        k=k,
        to_tensor=True,
    )

    if args.list_all:


        scan_ids_table = PrettyTable()
        scan_ids_table.hrules = ALL
        scan_ids_table.field_names = ["Available scan IDs"]
        for scan_id in test_dataset.scan_ids:
            scan_ids_table.add_row([scan_id])

        print(scan_ids_table)

    if args.scene:
        test_dataloader = DataLoader(test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=lambda x: x[0])

        for scan_idx, data_dict_list in enumerate(test_dataloader):
                
            # fetch data
            scan_id = test_dataset.scan_ids[scan_idx]
            if not scan_id == args.scene:
                continue

            points = test_dataset.data[scan_idx]['points'].astype(np.float32)
            seg_label = test_dataset.data[scan_idx]['seg_label']
            seg_label = test_dataset.nyu40_to_scannet[seg_label]

            from mvpnet.utils.visualize import visualize_labels
            visualize_labels(points, seg_label, window_name="Ground truth for {}".format(args.scene))




if __name__ == "__main__":
    main()