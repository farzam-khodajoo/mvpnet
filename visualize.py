import logging
from typing import List
import torch
import argparse
import pickle
import numpy as np
from pathlib import Path
from glob import glob
from plyfile import PlyData
import open3d as o3d
from mvpnet.utils.o3d_util import draw_point_cloud
from mvpnet.utils.visualize import label2color
from mvpnet.config.mvpnet_3d import cfg

from mvpnet.data.scannet_2d3d import ScanNet2D3DChunksTest, ScanNet2D3DChunks

from SETTING import ROOT_DIRECTORY, SCANNET_DIRECTORY

RESIZE_DATASET = Path(ROOT_DIRECTORY) / "scans_resize_160x120"

def raise_path_error(title, path=""):
    logging.error("{} {}".format(title, path))
    exit()


def read_ids():
    scannet_2d_dir = Path(ROOT_DIRECTORY) / "scans_resize_160x120"
    if not scannet_2d_dir.exists():
        print("[Error] path {} not found !".format(scannet_2d_dir))
        exit()

    dir_names = scannet_2d_dir.iterdir()
    dir_names: List[Path] = list(dir_names)  # convert to list from generator
    scan_ids = [name.name for name in dir_names]
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


def load_model(config):
    # the reason this dependencies are imported right here
    # is that model side requires compile CUDA libraries
    # which is impossible on non-nvidia device
    # so we import only when model is required
    from mvpnet.models.build import build_model_mvpnet_3d

    cfg.merge_from_file(str(config))
    model = build_model_mvpnet_3d(cfg)[0]
    model = model.cuda()
    model.eval()

    return model


def get_prediction_labels(predict):
    seg_logit = predict["seg_logit"].squeeze(0).cpu().detach().numpy().T

    return seg_logit


def predict_segmentation(model, data_batch: dict):
    # convert numpy into torch.tensor
    data_batch = {k: torch.tensor([v]) for k, v in data_batch.items()}
    # send to CUDA memory
    data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

    predict = model(data_batch)
    # send back to CPU
    predict = {k: v.cpu() for k, v in predict.items()}

    return get_prediction_labels(predict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="list available scan ids",
    )
    parser.add_argument(
        "--id",
        type=str,
        required=False,
        default=None,
        help="sample id from valdidation split",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        default=False,
        help="visualize predict of semantic segmentation",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=False,
        required=False,
        help="Save segmentation's output as pickle file",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=False,
        required=False,
        help="Load segmentation's output as pickle file",
    )
    parser.add_argument(
        "--less",
        action="store_true",
        default=False,
        help="avoid poping visualization window",
    )
    args = parser.parse_args()

    if args.list:
        id_list = read_ids()
        print("total ids: {}".format(len(id_list)))
        print("#" * 10)
        for scanid in id_list:
            print(scanid)
        exit()


    if args.list is None and args.id is None:
        print("[Error] either use --list or --id")

    if not args.id in read_ids():
        print("scan_id {} not found !".format(args.id))
        exit()

    point_cloud_path, labeled_cloud_path = query_3d_samples(
        str(Path(SCANNET_DIRECTORY) / args.id)
    )
    label_point_cloud, _, label = read_ply(labeled_cloud_path)

    # plot 3D space with labels
    if not args.less:
        print("Loading 3D space ..")
        result = draw_point_cloud(
            label_point_cloud, label2color(label, style="nyu40_raw")
        )
        o3d.visualization.draw_geometries([result], width=480, height=480)

    if args.predict or args.load is not False:
        model_config = Path.cwd() / "configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml"

        if not model_config.exists():
            raise_path_error("pn2ssg config file", model_config)

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
            cache_dir=cache_dir,
            image_dir=image_dir,
            split="val",
            to_tensor=True,
            num_rgbd_frames=4,
            chunk_size=(2, 2),
            chunk_margin=(10, 10),
        )

        search_id = [
            (idx, d)
            for idx, d in enumerate(test_dataset.data)
            if d["scan_id"] == args.id
        ]

        if len(search_id) == 0:
            print("scanid {} not found in dataset !".format(search_id))
            exit()

        scan_index, scan_context = search_id[0]

        if args.predict:

            logging.info("Loading model..")
            model = load_model(config=model_config)

            logging.info("Loading scene from dataset (preprocessing)")
            input_sample = test_dataset[scan_index][0]

            logging.info("Inferencing model")
            segmentation = predict_segmentation(model=model, data_batch=input_sample)
            scene_predicted_labels = np.argmax(segmentation, axis=1)

            logging.info("Creating point cloud")
            mvpnet_segmented_pt = draw_point_cloud(
                np.asarray(label_point_cloud), label2color(scene_predicted_labels)
            )

            o3d.visualization.draw_geometries(
                [mvpnet_segmented_pt], height=500, width=500
            )


if __name__ == "__main__":
    main()
