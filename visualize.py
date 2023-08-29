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

from SETTING import ROOT_DIRECTORY

RESIZE_DATASET = Path(ROOT_DIRECTORY) / "scans_resize_160x120"

def read_ids():
    scannet_2d_dir = Path(ROOT_DIRECTORY) / "2d_scannet"
    if not scannet_2d_dir.exists():
        print("[Error] path {} not found !".format(scannet_2d_dir))
        exit()

    dir_names = scannet_2d_dir.iterdir()
    dir_names: List[Path] = list(dir_names) # convert to list from generator
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="list available scan ids",
    )
    parser.add_argument(
        "--id", type=str, required=False, default=None, help="sample id from valdidation split"
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
        print("#"*10)
        for scanid in id_list:
            print(scanid)
        exit()

    if args.list is None and args.id is None:
        print("[Error] either use --list or --id")

    if not args.id in read_ids():
        print("scan_id {} not found !".format(args.id))
        exit()

    point_cloud_path, labeled_cloud_path = query_3d_samples(RESIZE_DATASET / args.id)
    label_point_cloud, _, label = read_ply(labeled_cloud_path)

    # plot 3D space with labels
    if not args.less:
        print("Loading 3D space ..")
        result = draw_point_cloud(
            label_point_cloud, label2color(label, style="nyu40_raw")
        )
        o3d.visualization.draw_geometries([result], width=480, height=480)

    if args.predict or args.load is not False:

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
            cache_dir=cache_dir,
            image_dir=image_dir,
            split="val",
            to_tensor=True,
            num_rgbd_frames=4,
            chunk_size=(2, 2),
            chunk_margin=(10, 10),
        )

        search_id = [d for d in test_dataset.data if d["scan_id"] == args.id]
        if len(search_id) == 0:
            print("scanid {} not found in dataset !".format(search_id))
            print(test_dataset.data)
            exit()

        scan_index = None

        for idx, dc in enumerate(test_dataset.data):
            if dc["scan_id"] == args.id:
                scan_index = idx
                print("index {} has been found".format(idx))

        if scan_index is None:
            print("scan index {} not found in dataset".format(args.id))
            exit()

        print("Fetch sample from dataset ..")
        input_sample = test_dataset.__getitem__(scan_index)
        input_sample = input_sample[0]
        chunk_ind = input_sample.pop('chunk_ind')

        print(input_sample.keys())

        if args.predict:
            print("Loading model ..")
            from mvpnet.models.build import build_model_mvpnet_3d
            from mvpnet.models.build import build_model_sem_seg_2d
            print("Building model ..")
            cfg.merge_from_file(str(model_config))
            model = build_model_mvpnet_3d(cfg)[0]
            model = model.cuda()
            model.eval()

            print("Predict ..")
            data_batch = {k: torch.tensor([v]) for k, v in input_sample.items()}
            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}
            predict = model(data_batch)
            # move from CUDA to cpu
            predict = {k: v.cpu() for k, v in predict.items()}

        if args.load is not False:
            if not Path(args.load).exists():
                print("loading failed ! path {} not found".format(args.load))

            print("Loading pickle from {}".format(args.load))

            with open(args.load, "rb") as handle:
                predict = pickle.load(handle)

        seg_logit = predict["seg_logit"].squeeze(0).cpu().detach().numpy().T
        seg_logit = seg_logit[:len(chunk_ind)]

        print("segmentation logit: ", seg_logit.shape)
        print("segmentation points: ", np.unique(seg_logit))
        print("segmentation argmax: ", np.argmax(seg_logit, axis=1).shape)
        print("label shape: ", input_sample["points"].shape)

        chunk_points = input_sample["points"]
        chunk_points = chunk_points.reshape((chunk_points.shape[1], 3))
        logit_labels = np.argmax(seg_logit, axis=1)

        print("chunks: {} -> {}".format(label_point_cloud.shape, chunk_points.shape))
        print("labels: {} -> {}".format(label.shape, logit_labels.shape))
        print("chunks: ", chunk_ind.shape)

        if not args.less:
            print("Loading 3D space ..")
            result = draw_point_cloud(
                chunk_points, label2color(logit_labels, style="nyu40_raw")
            )
            o3d.visualization.draw_geometries([result], width=480, height=480)


        if not args.save is False:
            if not Path(args.save).parent.exists():
                print("saving path {} is invalid !".format(args.save))
                exit()

            with open(args.save, "wb") as handle:
                pickle.dump(predict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("output saved into {}".format(args.save))


if __name__ == "__main__":
    main()
