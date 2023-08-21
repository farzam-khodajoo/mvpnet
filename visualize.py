import argparse
import numpy as np
from pathlib import Path
from glob import glob
from plyfile import PlyData
import open3d as o3d
from mvpnet.utils.o3d_util import draw_point_cloud
from mvpnet.utils.visualize import label2color
from mvpnet.config.mvpnet_3d import cfg

# from mvpnet.models.build import build_model_mvpnet_3d
from mvpnet.data.scannet_2d3d import ScanNet2D3DChunksTest

from SETTING import ROOT_DIRECTORY

RESIZE_DATASET = Path(ROOT_DIRECTORY) / "scans_resize_160x120"
META_DIRECTORY = Path.cwd() / "mvpnet" / "data" / "meta_files" / "scannetv2_val.txt"


def read_ids():
    split_filename = META_DIRECTORY
    with open(split_filename, "r") as f:
        scan_ids = [line.rstrip() for line in f]
    return scan_ids


def query_3d_samples(sample_directory):
    scan_id = Path(sample_directory).name
    point_cloud = Path(sample_directory) / f"{scan_id}_vh_clean_2.ply"
    labeled_cloud = Path(sample_directory) / f"{scan_id}_vh_clean_2.labels.ply"
    return point_cloud, labeled_cloud


def read_ply(filename):
    ply_data = PlyData.read(filename)
    vertex = ply_data["vertex"]
    x = np.asarray(vertex["x"])
    y = np.asarray(vertex["y"])
    z = np.asarray(vertex["z"])
    points = np.stack([x, y, z], axis=1)

    r = np.asarray(vertex["red"])
    g = np.asarray(vertex["green"])
    b = np.asarray(vertex["blue"])
    colors = np.stack([r, g, b], axis=1)

    label = np.asarray(vertex["label"])

    return points, colors, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id", type=str, required=True, help="sample id from valdidation split"
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        default=False,
        help="visualize predict of semantic segmentation",
    )
    parser.add_argument(
        "--less",
        action="store_true",
        default=False,
        help="avoid poping visualization window",
    )
    args = parser.parse_args()

    if not args.id in read_ids():
        print("scan_id {} not found !".format(args.id))
        exit()

    # plot 3D space with labels
    if not args.less:
        print("Loading 3D space..")
        point_cloud_path, labeled_cloud_path = query_3d_samples(RESIZE_DATASET / args.id)
        label_point_cloud, _, label = read_ply(labeled_cloud_path)
        result = draw_point_cloud(
            label_point_cloud, label2color(label, style="nyu40_raw")
        )
        o3d.visualization.draw_geometries([result], width=480, height=480)

    if args.predict:
        print("Loading Model..")
        model_config = str(
            Path.cwd() / "configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml"
        )

        if not Path(model_config).exists():
            print("path {} not found !".format(model_config))
            exit()
        cfg.merge_from_file(model_config)

        # model, loss_fn, train_metric, val_metric = build_model_mvpnet_3d(cfg)

        print("Loading dataset ..")
        cache_dir = Path(ROOT_DIRECTORY) / "pickles"
        image_dir = Path(ROOT_DIRECTORY) / "scans_resize_160x120"

        if not cache_dir.exists():
            print("folder {} not found !".format(cache_dir))
            exit()

        if not image_dir.exists():
            print("folder {} not found !".format(image_dir))
            exit()

        test_dataset = ScanNet2D3DChunksTest(
            cache_dir=cache_dir, image_dir=image_dir, split="val", to_tensor=True
        )

        search_id = [d for d in test_dataset.data if d["scan_id"] == args.id]
        if len(search_id) == 0:
            print("scanid {} not found in dataset !".format(search_id))



if __name__ == "__main__":
    main()
