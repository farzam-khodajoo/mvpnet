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
        "--image",
        type=str,
        required=False,
        help="the target scene id",
    )
    parser.add_argument(
        "--load",
        type=str,
        required=False,
        help="load segmentation from picke file",
    )
    parser.add_argument(
        "--save",
        type=str,
        required=False,
        help="save segmentation result into pickle",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=False,
        help="only show segmentation for specific label",
    )

    args = parser.parse_args()

    pickle_directory = Path(ROOT_DIRECTORY)/"pickles"
    scan_directory = Path(ROOT_DIRECTORY)/"scans_resize_160x120"

    if not pickle_directory.exists():
        raise_path_error("Pickle directory", pickle_directory)

    if not scan_directory.exists():
        raise_path_error("Scan directory", pickle_directory)


    if args.image:
        image = Path(args.image)
        if not image.exists():
            raise_path_error("image", image)

        if args.load:
            prediction_labels_context = Pickle.load_from_pickle(args.load)

        else:
            from utils.inference import inference_2d
            prediction_labels_context = inference_2d(image_path=image)

        all_class_ids = Path.cwd() / "mvpnet/data/meta_files/labelids.txt"
        label_dict = load_class_mapping(all_class_ids)   
        labels_in_scene = np.unique(prediction_labels_context)

        all_label_names = get_available_labels(labels_in_scene, label_dict)

        label_table = PrettyTable()
        label_table.hrules = ALL
        label_table.field_names = ["Labels"]
        for label_name in all_label_names:
            label_table.add_row([label_name])

        print(label_table)

        logging.info("Loading Segmentation window..")
        from mvpnet.utils.visualize import label2color

        width_size, height_size = prediction_labels_context.shape
        prediction_labels = np.rot90(prediction_labels_context, k=1)
        flipped_prediction_labels = np.flip(prediction_labels, axis=1)
        flipped_prediction_labels = np.flip(prediction_labels, axis=0)
        import matplotlib.pyplot as plt
        plt.imshow(flipped_prediction_labels)
        plt.show()
        
        if args.save:
            if not Path(args.save).parent.exists():
                raise_path_error("Parent folder", args.save)
            
            Pickle.save_into_pickle(prediction_labels_context, path=Path(args.save))

if __name__ == "__main__":
    main()