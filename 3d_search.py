VOXEL_SIZE = 0.0001

from collections import OrderedDict
import os.path as osp
import csv
import logging
from typing import List
import argparse
import pickle
import torch
from pathlib import Path
from glob import glob
from SETTING import ROOT_DIRECTORY, SCANNET_DIRECTORY
from mvpnet.utils.o3d_util import draw_point_cloud
from mvpnet.utils.visualize import label2color
from mvpnet.data.scannet_2d3d import ScanNet2D3DChunksTest, ScanNet2D3DChunks
from common.utils.checkpoint import CheckpointerV2
from mvpnet.config.mvpnet_3d import cfg
from custome_sampler import get_input_batch_sample

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm
from prettytable import PrettyTable


DEPTH_SCALE_FACTOR = 100.0

logging.basicConfig(level=logging.INFO)


def raise_path_error(title, path=""):
    logging.error("{} {}".format(title, path))
    exit()


def load_from_pickle(path: str):
    if not Path(path).exists():
        raise_path_error("pickle file", path)

    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_into_pickle(object, path: str):
    with open(path, "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_ids(dir=None):
    scannet_3d_dir = Path(SCANNET_DIRECTORY) if dir is None else Path(dir)
    if not scannet_3d_dir.exists():
        raise_path_error("3D scannet directory", scannet_3d_dir)

    dir_names = scannet_3d_dir.iterdir()
    dir_names: List[Path] = list(dir_names)  # convert to list from generator
    scan_ids = [name.name for name in dir_names]
    return scan_ids


def get_scenes(dir=None):
    scene_dir = Path(SCANNET_DIRECTORY) if dir is None else Path(dir)
    scan_ids = read_ids(dir)
    return [scene_dir / scanid for scanid in scan_ids]


def load_class_mapping(filename):
    id_to_class = OrderedDict()
    with open(filename, "r") as f:
        for line in f.readlines():
            class_id, class_name = line.rstrip().split("\t")
            id_to_class[int(class_id)] = class_name
    return id_to_class


def read_label_mapping(
    filename, label_from="raw_category", label_to="nyu40id", as_int=False
):
    """Read the mapping of labels from the given tsv"""
    assert osp.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            key = row[label_from]
            if as_int:
                key = int(key)
            mapping[key] = int(row[label_to])
    return mapping


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


def get_prediction_segmentation_from_pickle(path):
    logits = load_from_pickle(path)
    return logits


def load_cache(path):
    if not Path(path).exists():
        raise_path_error("cache file", path)

    with open(path, "rb") as cache:
        print("[Info] Loading cache file")
        cache_data = pickle.load(cache)
        return cache_data

def to_cpu(obj):
    if type(obj) == dict:
        return {k: v.cpu() for k, v in obj.items()}
    
    return obj.cpu()

def plot_2d3d_fusion(model, data_batch):
    import torch
    from torch import nn
    import numpy as np

    from common.nn import SharedMLP
    from mvpnet.ops.group_points import group_points

    # convert numpy into torch.tensor
    data_batch = {k: torch.tensor([v]) for k, v in data_batch.items()}
    # send to CUDA memory
    data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

    # (batch_size, num_views, 3, h, w)
    images = data_batch['images']
    b, nv, _, h, w = images.size()
    # collapse first 2 dimensions together
    images = images.reshape([-1] + list(images.shape[2:]))

    # 2D network
    preds_2d = model.net_2d({'image': images})
    feature_2d = preds_2d['feature']  # (b * nv, c, h, w)

    # torch.Size([1, 20, 120, 160]) 
    #print(preds_2d["seg_logit"].size())
    #print(feature_2d.size())

    # unproject features
    knn_indices = data_batch['knn_indices']  # (b, np, k)
    feature_2d = feature_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
    feature_2d = feature_2d.reshape(b, -1, nv * h * w)
    feature_2d = group_points(feature_2d, knn_indices)  # (b, c, np, k)

    # unproject depth maps
    with torch.no_grad():
        image_xyz = data_batch['image_xyz']  # (b, nv, h, w, 3)
        image_xyz = image_xyz.permute(0, 4, 1, 2, 3).reshape(b, 3, nv * h * w)
        image_xyz = group_points(image_xyz, knn_indices)  # (b, 3, np, k)

    # 2D-3D aggregation
    points = data_batch['points']
    feature_2d3d = model.feat_aggreg(image_xyz, points, feature_2d)

    # torch.Size([1, 64, 276996])
    # print(feature_2d3d.size())

    # 3D network
    preds_3d = model.net_3d({'points': points, 'feature': feature_2d3d})
    preds = preds_3d

    out_features = {}
    out_features["2d_feature"] = to_cpu(feature_2d)
    out_features["2d_logit"] = to_cpu(preds_2d["seg_logit"])
    out_features["image_xyz"] = to_cpu(image_xyz)
    out_features["aggreg"] = to_cpu(feature_2d3d)
    out_features["3d_logit"] = to_cpu(preds)

    return out_features

def search_scanid(cache_data, scanid):
    id_query = [d for d in cache_data if d["scan_id"] == scanid]
    if len(id_query) == 0:
        raise_path_error("query scan id", scanid)

    return id_query[0]


def read_pc_from_ply(filename, return_color=False, return_label=False):
    """Read point clouds from ply files"""
    ply_data = PlyData.read(filename)
    vertex = ply_data["vertex"]
    x = np.asarray(vertex["x"])
    y = np.asarray(vertex["y"])
    z = np.asarray(vertex["z"])
    points = np.stack([x, y, z], axis=1)
    pc = {"points": points}
    if return_color:
        r = np.asarray(vertex["red"])
        g = np.asarray(vertex["green"])
        b = np.asarray(vertex["blue"])
        colors = np.stack([r, g, b], axis=1)
        pc["colors"] = colors
    if return_label:
        label = np.asarray(vertex["label"])
        pc["label"] = label
    return pc


def unproject(k, depth_map, mask=None):
    print(max(np.unique(depth_map)))
    if mask is None:
        # only consider points where we have a depth value
        mask = depth_map > 0
    # create xy coordinates from image position
    v, u = np.indices(depth_map.shape)
    v = v[mask]
    u = u[mask]
    depth = depth_map[mask].ravel()
    uv1_points = np.stack([u, v, np.ones_like(u)], axis=1)
    points_3d_xyz = (np.linalg.inv(k[:3, :3]).dot(uv1_points.T) * depth).T
    return points_3d_xyz


def create_bounding_box(pts):
    # Get the minimum and maximum coordinates of the point cloud
    min_bound, max_bound = pts.get_min_bound(), pts.get_max_bound()

    # Define the bounding box corners
    corners = np.array(
        [
            [min_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
        ]
    )

    # Create a line set to represent the bounding box
    lines = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Set line colors (optional)
    line_set.colors = o3d.utility.Vector3dVector(
        np.array([[1, 0, 0] for _ in range(len(lines))])
    )
    return line_set

def set_color_for_overlaps_in_scene(scene_pts, scene_colors, overlap_pts, color):
    """Change a set of point in scene to specific color,
        this is used along with bounding box for segmented object
    """
    points_insize_bbox = np.all((overlap_pts.get_min_bound() <= scene_pts.points) & (scene_pts.points <= overlap_pts.get_max_bound()), axis=1)
    scene_colors[points_insize_bbox] = color
    scene_pts.colors = o3d.utility.Vector3dVector(scene_colors)

    return scene_pts


def main():
    global SCANNET_DIRECTORY
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default=SCANNET_DIRECTORY,
        help="path to scans/ dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="sample which should be searched on dataset",
    )
    parser.add_argument(
        "--mvpnet",
        action="store_true",
        default=False,
        help="using mvpnet for fusion segmentation",
    )
    parser.add_argument(
        "--pickle",
        type=str,
        required=False,
        default=None,
        help="path to cache .pickle file",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=False,
        default=None,
        help="label to be segmented from mvpnet",
    )
    parser.add_argument(
        "--save-segmentation-as-pickle",
        type=str,
        required=False,
        default=None,
        help="save model logit into pickle",
    )
    parser.add_argument(
        "--icp-threshold",
        type=float,
        required=False,
        default=0.5,
        help="threshold parameter for ICP",
    )
    parser.add_argument(
        "--max-icp-iter",
        type=int,
        required=False,
        default=5_000,
        help="max iteration for ICP",
    )
    parser.add_argument(
        "--knn-radius",
        type=float,
        required=False,
        default=0.5,
        help="max iteration for ICP",
    )
    parser.add_argument(
        "--knn-nn",
        type=float,
        required=False,
        default=1,
        help="max iteration for ICP",
    )
    parser.add_argument(
        "--compare-mvpnet-labels",
        action="store_true",
        required=False,
        default=None,
        help="log difference between predicted labels and real one",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        required=False,
        default=None,
        help="avoid searching 3D search",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="visualize saved report",
    )
    parser.add_argument(
        "--rmse-max",
        action="store_true",
        default=False,
        help="sort candidates based on maximum rmse",
    )
    args = parser.parse_args()

    if args.dataset is not None:
        if not Path(args.dataset).exists():
            logging.error("Dataset directory {} does not exists".format(args.dataset))
            exit()

        len_samples = glob(str(Path(args.dataset) / "*"))
        logging.info("Loading {} samples from dataset {}".format(len(len_samples), args.dataset))
        SCANNET_DIRECTORY = str(Path(args.dataset))

    target_dir = Path(args.target)
    save_dir = Path(args.target) / "report"

    if not target_dir.exists():
        raise_path_error("target folder", target_dir)

    # create report directory
    save_dir.mkdir(exist_ok=True)

    if args.report:
        scene_ply = glob(str(save_dir / "*.ply"))
        bbox = glob(str(save_dir / "*.npy"))

        if len(scene_ply) < 1:
            raise_path_error("report folder's .ply")

        if len(bbox) < 1:
            raise_path_error("report folder's .npy")

        pc = read_pc_from_ply(scene_ply[0])
        pts_vis = draw_point_cloud(pc["points"]).voxel_down_sample(voxel_size=VOXEL_SIZE)

        bbox = np.load(bbox[0])
        overlap_point_cloud = draw_point_cloud(np.asarray(pts_vis.points)[bbox[:]])
        bbox = create_bounding_box(overlap_point_cloud)
        logging.info("Loading scene")
        o3d.visualization.draw_geometries([bbox, pts_vis], height=500, width=500)
        exit()

    # check color sample
    sample_color_glob = glob(str(target_dir / "*.jpg"))
    if len(sample_color_glob) < 1:
        raise_path_error("color.jpg file")

    sample_color = sample_color_glob[0]

    # check depth sample
    sample_depth_glob = glob(str(target_dir / "*.png"))
    if len(sample_depth_glob) < 1:
        raise_path_error("color.jpg file")

    sample_depth = sample_depth_glob[0]

    color_sample_id = Path(sample_color).name.split(".jpg")[0]
    depth_sample_id = Path(sample_depth).name.split(".png")[0]

    if color_sample_id != depth_sample_id:
        logging.error("color.jpg id and depth.jpg id mismatch !")
        logging.error(
            "color: {} --- depth: {}".format(color_sample_id, depth_sample_id)
        )
        exit()

    sample_pose = target_dir / "{}.txt".format(depth_sample_id)
    if not sample_pose.exists():
        raise_path_error("pose txt file", sample_pose)

    sample_intrinsic_depth = target_dir / "intrinsic_depth.txt"
    if not sample_intrinsic_depth.exists():
        raise_path_error("intrinsic file", sample_intrinsic_depth)

    # load image
    color_image = Image.open(sample_color)
    color = np.array(color_image)
    print("rgb", color.shape)

    # loading instrinsic data
    cam_matrix = np.loadtxt(sample_intrinsic_depth, dtype=np.float32)

    # load depth image
    depth_image = Image.open(sample_depth)
    depth = np.asarray(depth_image, dtype=np.float32) / DEPTH_SCALE_FACTOR

    # load pose data
    pose = np.loadtxt(sample_pose)

    # unporoject 2D to 3D
    unproj_pts = unproject(cam_matrix, depth)
    unproj_pts = pose[:3, :3].dot(unproj_pts[:, :3].T).T + pose[:3, 3]
    masked_unproj_pts = None


    logging.info("project cloud shape: {}".format(unproj_pts.shape))

    if args.mvpnet:
        # check if config file exists
        model_config = (
            Path.cwd() / "configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml"
        )
        if not model_config.exists():
            raise_path_error("pn2ssg config file", model_config)

        # pre-process data
        logging.info("Loading batch sample")
        data = get_input_batch_sample(
            points=unproj_pts,
            color=color_image,
            depth=depth_image,
            pose=pose,
            cam_matrix=cam_matrix,
            depth_scale_factor=1000.0,
        )

        save_into_pickle(data, "batch.data.pickle")

        # create label mapping
        all_class_ids = Path.cwd() / "mvpnet/data/meta_files/labelids.txt"
        if not all_class_ids.exists():
            raise_path_error("file", all_class_ids)

        labels_combined_path = (
            Path.cwd() / "mvpnet/data/meta_files/scannetv2-labels.combined.tsv"
        )
        if not labels_combined_path.exists():
            raise_path_error("file", labels_combined_path)

        raw_to_nyu40_mapping = read_label_mapping(
            labels_combined_path, label_from="id", label_to="nyu40id", as_int=True
        )

        raw_to_nyu40 = np.zeros(max(raw_to_nyu40_mapping.keys()) + 1, dtype=np.int64)

        for key, value in raw_to_nyu40_mapping.items():
            raw_to_nyu40[key] = value

        label_mapping = load_class_mapping(all_class_ids)
        assert len(label_mapping) == 20
        nyu40_to_scannet = np.full(shape=41, fill_value=-100, dtype=np.int64)
        nyu40_to_scannet[list(label_mapping.keys())] = np.arange(len(label_mapping))
        scannet_to_nyu40 = np.array(list(label_mapping.keys()) + [0], dtype=np.int64)
        raw_to_scannet = nyu40_to_scannet[raw_to_nyu40]
        label_map_names = tuple(label_mapping.values())
        label_map_ids = [id_ for id_, classname in label_mapping.items()]

        fusion_features = None

        if args.pickle is not None:
            logging.info("Loading segmentation from pickle")
            segmentation = get_prediction_segmentation_from_pickle(args.pickle)

            # load feature's pickle file if exists
            feature_path = Path(args.pickle)
            feature_header_name = feature_path.name
            feature_dirname = feature_path.parent
            feature_path = Path(feature_dirname) / "features.{}".format(feature_header_name)

            if feature_path.exists():
                logging.info("Loading features from {}".format(feature_path))
                fusion_features = load_from_pickle(feature_path)
            else:
                logging.warning("Skip loading feature's pickle as {} not found".format(feature_path))


        else:
            logging.info("Loading model")
            model = load_model(config=model_config)

            logging.info("Loading weights")
            output_directory = (
                Path.cwd() / "outputs/scannet/mvpnet_3d_unet_resnet34_pn2ssg"
            )

            if not output_directory.exists() or not output_directory.is_dir():
                logging.error(
                    "model checkpoint direcotry {} does not exists".format(
                        output_directory
                    )
                )
                exit()

            logging.info("Loading weights from checkpointer")
            checkpointer = CheckpointerV2(model, save_dir=output_directory)
            if not checkpointer.has_checkpoint():
                logging.error("No checkpoint has been found")

            # this will resume and load last checkpoint
            checkpointer.load(None, resume=True)

            logging.info("Processing segmentation")
            segmentation = predict_segmentation(model=model, data_batch=data)
            logging.info("Processing features")
            fusion_features = plot_2d3d_fusion(model=model, data_batch=data)

        # save model output into pickle
        if args.save_segmentation_as_pickle is not None:
            if args.pickle is not None:
                logging.error("avoid re-saving pickle")
                exit()

            logging.info(
                "Save segmentation output into {}".format(
                    args.save_segmentation_as_pickle
                )
            )
            # save model segmentation output into pickle file
            save_into_pickle(segmentation, args.save_segmentation_as_pickle)
            # save model features into pickle file
            feature_path = Path(args.save_segmentation_as_pickle)
            feature_header_name = feature_path.name
            feature_dirname = feature_path.parent
            feature_path = Path(feature_dirname) / "features.{}".format(feature_header_name)
            logging.info(
                "Save model's features into {}".format(
                    feature_path
                )
            )
            save_into_pickle(fusion_features, feature_path)

        scene_predicted_labels = np.argmax(segmentation, axis=1)

        scene_prediction_nyu40 = scannet_to_nyu40[scene_predicted_labels]
        # print(np.unique(scene_prediction_nyu40))
        predicted_labels = np.unique(scene_predicted_labels)
        logging.info("Label IDs found in prediction {}".format(predicted_labels))
        predicted_classnames = [label_map_names[int(idx)] for idx in predicted_labels]
        logging.info("Labels found in prediction: {}".format(predicted_classnames))

        # pop window and show differne between prediction and labels
        if args.compare_mvpnet_labels is not None:
            labels = Path(args.target) / "addition" / "{}.png".format(depth_sample_id)
            if labels.exists():
                # load labeled image and segment with only valid labels
                label_image = Image.open(labels)
                label_np = np.array(label_image)
                bad_label_ind = np.logical_or(label_np < 0, label_np > 40)
                label_np[bad_label_ind] = 0
                segmentaion_labels = nyu40_to_scannet[label_np]

                logging.info("Plotting label images")
                plt.figure(figsize=(7, 4))
                # label image including invalid images
                plt.subplot(121)
                plt.imshow(label_np)
                plt.title("Original")

                plt.subplot(122)
                plt.imshow(segmentaion_labels)
                plt.title("Valid only (exclude -100)")

                plt.tight_layout()
                plt.show()


            # loading Resnet segmentation
            logging.info("Loading 2D Resnet segmentation")
            resnet_output = fusion_features["2d_logit"].squeeze(0).detach().numpy().T
            resnet_output = np.argmax(resnet_output, axis=2)
            width_size, height_size = resnet_output.shape
            resnet_output = np.rot90(resnet_output, k=1)
            flipped_resnet_output = np.flip(resnet_output, axis=1)
            flipped_resnet_output = np.flip(resnet_output, axis=0)
            plt.imshow(flipped_resnet_output)
            plt.show()

            logging.info("Loading depth map projection")
            image_xyz = fusion_features["image_xyz"].squeeze(0).detach().numpy()
            image_xyz_pts = []
            for idx in range(len(image_xyz)):
                image_xyz_pts.append(draw_point_cloud(image_xyz[idx]))

            o3d.visualization.draw_geometries(image_xyz_pts, width=500, height=500)

            mvpnet_segmented_pt = draw_point_cloud(
                unproj_pts, label2color(scene_predicted_labels)
            ).voxel_down_sample(voxel_size=VOXEL_SIZE)

            logging.info("Loading segmented view from mvpnet")
            o3d.visualization.draw_geometries(
                [mvpnet_segmented_pt], height=500, width=500
            )

        if args.label is not None:
            try:
                index_of_classname = label_map_names.index(args.label) - 1
            except ValueError:
                logging.error(
                    "Classname {} not found in {}".format(args.label, label_map_names)
                )
                exit()

            logging.info(
                "Slicing projection with mask label '{}' and index of {}".format(
                    args.label, index_of_classname + 2
                )
            )
            choosen_label = index_of_classname
            label_mask = np.where(scene_predicted_labels == choosen_label)
            masked_unproj_pts = unproj_pts[label_mask]

        if args.no_search:
            logging.info("Exit program after --no-search")
            exit()

    # visualize projection before proceed search
    logging.info("Loading pre-view of original projection")
    unproj_pts_vis = draw_point_cloud(unproj_pts)

    o3d.visualization.draw_geometries([unproj_pts_vis], height=500, width=500)
    if masked_unproj_pts is not None:
        logging.info("Loading pre-view of masked projection")
        masked_unproj_pts_vis = draw_point_cloud(masked_unproj_pts, colors=color/255.)
        o3d.visualization.draw_geometries(
            [masked_unproj_pts_vis], height=500, width=500
        )
        logging.info("previous projection size: {}".format(unproj_pts.shape))
        unproj_pts = masked_unproj_pts
        logging.info("New masked projection size: {}".format(unproj_pts.shape))


    param_table = PrettyTable()
    param_table.field_names = ["Param", "value"]
    param_table.add_row(["KDTreeFlann radius", args.knn_radius])
    param_table.add_row(["KDTreeFlann nn", args.knn_nn])
    param_table.add_row(["ICP threshold", args.icp_threshold])
    param_table.add_row(["ICP maximum iteration", args.max_icp_iter])

    print(param_table)
    # container for knn distance results
    knn_distances = {}

    all_scenes = get_scenes(args.dataset)
    for idx, scene in enumerate(all_scenes):
        # check if folder exists
        if not scene.exists():
            raise_path_error("scene", scene)

        logging.info("Loading scene from {} ({}/{})".format(scene, idx+1, len(all_scenes)))

        sample_id = scene.name
        sample_ply_scene_path = scene / "{}_vh_clean_2.ply".format(sample_id)
        pc = read_pc_from_ply(sample_ply_scene_path, return_color=True)
        pts_vis = draw_point_cloud(pc["points"])

        unproj_pts_vis = draw_point_cloud(unproj_pts)

        pcd_tree = o3d.geometry.KDTreeFlann(pts_vis)
        overlaps = np.zeros([len(np.asarray(pts_vis.points))], dtype=bool)

        for idx in tqdm(range(len(unproj_pts))):
            # find a neighbor that is at most 1cm away
            found, idx_point, dist = pcd_tree.search_hybrid_vector_3d(
                unproj_pts[idx, :3], args.knn_radius, args.knn_nn
            )
            if found:
                overlaps[idx_point] = True

        # check if any overlap have been found
        if True in np.unique(overlaps):
            # Define ICP parameters
            threshold = 0.5  # Maximum correspondence point-pair distance
            trans_init = np.identity(4)  # Initial transformation (identity matrix)
            max_iter = 5_000  # Maximum number of ICP iterations
            overlap_base_pts_vis = draw_point_cloud(
                np.asarray(pts_vis.points)[overlaps[:]]
            )

            reg_p2p = o3d.registration.registration_icp(
                unproj_pts_vis,
                overlap_base_pts_vis,
                args.icp_threshold,
                trans_init,
                o3d.registration.TransformationEstimationPointToPoint(),
                o3d.registration.ICPConvergenceCriteria(max_iteration=args.max_icp_iter),
            )

            alignment_error = reg_p2p.inlier_rmse

            knn_distances.update(
                {alignment_error: {"overlap": overlaps, "id": sample_id}}
            )

    if len(knn_distances) < 1:
        logging.warning("No similarity have been found.")
        exit()

    logging.info("Sorting results")
    query_distances = [dis for dis in knn_distances.keys()]
    min_distance = max(query_distances) if args.rmse_max else min(query_distances)

    nearest_result = knn_distances[min_distance]
    result_scan_id = nearest_result["id"]
    result_overlap = nearest_result["overlap"]
    logging.info("Found #1 first candidate {} with RMSE of {}".format(result_scan_id, min_distance))

    query_distances.sort(reverse=True if args.rmse_max else False)
    second_min_distance = query_distances
    second_min_distance = second_min_distance[1] # [0] will be first

    if query_distances[0] != min_distance:
        logging.error("Result sorting mismatching")
        exit()

    second_nearest_result = knn_distances[second_min_distance]
    second_scan_id = second_nearest_result["id"]
    logging.info("Found #2 second candidate {} with RMSE of {}".format(second_scan_id, second_min_distance))


    if args.dataset is not None:
        scene = (
            Path(args.dataset)
            / result_scan_id
            / "{}_vh_clean_2.ply".format(result_scan_id)
        )

    else:
        scene = (
            Path(SCANNET_DIRECTORY)
            / result_scan_id
            / "{}_vh_clean_2.ply".format(result_scan_id)
        )
    

    target_fixed_color = [0.7, 0.2, 0.1]
    scene_ply = read_pc_from_ply(scene, return_color=True)
    scene_colors = scene_ply["colors"] / 255.
    scene_point_cloud = draw_point_cloud(scene_ply["points"], colors=scene_colors)
    overlap_point_cloud = draw_point_cloud(
        np.asarray(scene_point_cloud.points)[result_overlap[:]], colors=target_fixed_color
    )

    bbox = create_bounding_box(overlap_point_cloud)

    logging.info("Result shown from scene {}".format(result_scan_id))
    o3d.visualization.draw_geometries([bbox, set_color_for_overlaps_in_scene(
        scene_pts=scene_point_cloud,
        scene_colors=scene_colors,
        overlap_pts=overlap_point_cloud,
        color=target_fixed_color
    )], height=500, width=500)

    logging.info("Writing into {}".format(save_dir))
    o3d.io.write_point_cloud(
        str(save_dir / "{}.ply".format(result_scan_id)), scene_point_cloud
    )
    np.save(save_dir / "{}_lines".format(result_scan_id), result_overlap)

    # saving rmse results into report/rmse.csv
    logging.info("Getting RMSE report..")
    import pandas as pd

    rmse_report_context = {
        "space_id": [],
        "rmse_score": []
    }

    for rmse_score in knn_distances.keys():
        room_name = knn_distances[rmse_score]["id"]
        rmse_report_context["space_id"].append(room_name)
        rmse_report_context["rmse_score"].append(rmse_score)


    rmse_df = pd.DataFrame(rmse_report_context)
    print(rmse_df.to_markdown())

    rmse_report_save_destination = save_dir / "rmse.csv"
    logging.info("Sasving RMSE report into {}".format(rmse_report_save_destination))
    rmse_df.to_csv(rmse_report_save_destination)

    # save center xyz into file
    with open(save_dir / "position.txt", "w") as file:
        centroid = overlap_point_cloud.get_center()
        logging.info("Saving {} coordinate into {}".format(centroid, save_dir / "position.txt"))
        file.write(f"{centroid[0]}\n{centroid[1]}\n{centroid[2]}")


    """
        if args.mvpnet:
        logging.info("Loading MVPNet segmentation for entire scene {}".format(result_scan_id))
        cache_dir = Path(ROOT_DIRECTORY) / "pickles"
        image_dir = Path(ROOT_DIRECTORY) / "scans_resize_160x120"
        if not cache_dir.exists(): raise_path_error("cache file", cache_dir)
        if not image_dir.exists(): raise_path_error("image directory", image_dir)

        logging.info("Loading dataset")
        test_dataset = ScanNet2D3DChunksTest(
            cache_dir=cache_dir,
            image_dir=image_dir,
            split="test",
            to_tensor=True,
            num_rgbd_frames=4,
            chunk_size=(2, 2),
            chunk_margin=(10, 10),
        )

        test_dataset.scan_ids = read_ids()

        search_id = [(idx, d) for idx, d in enumerate(test_dataset.data) if d["scan_id"] == result_scan_id]
        if len(search_id) == 0:
            logging.error("Scene {} not found in test_dataset".format(result_scan_id))
            exit()

        scan_index, scan_context = search_id[0]

        logging.info("Loading scene from dataset (preprocessing)")
        input_sample = test_dataset[scan_index][0]
        logging.info("Inferencing model")
        segmentation = predict_segmentation(model=model, data_batch=input_sample)
        scene_predicted_labels = np.argmax(segmentation, axis=1)

        logging.info("Creating point cloud")
        mvpnet_segmented_pt = draw_point_cloud(
            np.asarray(scene_point_cloud.points), label2color(scene_predicted_labels)
        ).voxel_down_sample(voxel_size=VOXEL_SIZE)

        o3d.visualization.draw_geometries(
            [bbox, overlap_point_cloud], height=500, width=500
        )
        
    """
    
        

if __name__ == "__main__":
    main()
