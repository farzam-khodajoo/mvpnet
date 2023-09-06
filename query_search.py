import torch
import open3d as o3d
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from glob import glob
from PIL import Image
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
from prettytable import PrettyTable
import argparse
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as CM, ConfusionMatrixDisplay
from SETTING import ROOT_DIRECTORY, SCANNET_DIRECTORY


from common.utils.checkpoint import CheckpointerV2
from mvpnet.config.mvpnet_3d import cfg
from custome_sampler import get_input_batch_sample
from mvpnet.utils.o3d_util import draw_point_cloud

# configure logging
import logging

logging.basicConfig(level=logging.INFO)

SCANNET_2D_DIRECTORY = Path(ROOT_DIRECTORY) / "2d_scannet"


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


def check_path(path: Path):
    """Exit if path does not exist"""
    if not Path(path).exists():
        logging.error("Path {} does not exists !".format(path))
        exit()


def unproject(k, depth_map, mask=None):
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


def calculate_confusion(true_labels, predictions, labels):
    return CM(true_labels, predictions, labels=labels)


class Query2D:
    """query all 2D image samples from single scene and loop through"""

    def __init__(self, scannet_2d_dir: Path, depth_scale: float = 100.0) -> None:
        self.scannet_2d_dir = Path(scannet_2d_dir)
        check_path(self.scannet_2d_dir)

        self.__depth_scale = depth_scale

        self.scans_glob = glob(str(self.scannet_2d_dir / "*"))
        logging.info(
            "Found {} scans in {}".format(len(self.scans_glob), self.scannet_2d_dir)
        )

        self.colors = glob(str(self.scannet_2d_dir / "*" / "color" / "*.jpg"))
        self.depths = glob(str(self.scannet_2d_dir / "*" / "depth" / "*.png"))
        self.poses = glob(str(self.scannet_2d_dir / "*" / "pose" / "*.txt"))
        self.labels = glob(str(self.scannet_2d_dir / "*" / "label" / "*.png"))
        self.intrinsics = glob(
            str(self.scannet_2d_dir / "*" / "intrinsic" / "intrinsic_depth.txt")
        )

        logging.info("Total colors found: {}".format(len(self.colors)))
        logging.info("Total depths found: {}".format(len(self.depths)))
        logging.info("Total poses found: {}".format(len(self.poses)))
        logging.info("Total labels found: {}".format(len(self.labels)))
        logging.info("Total intrinsics found: {}".format(len(self.intrinsics)))

        self.ignore_value = -100
        self.nyu40_to_scannet = np.full(
            shape=41, fill_value=self.ignore_value, dtype=np.int64
        )

        self.scannet_classes_path = Path.cwd() / "mvpnet/data/meta_files/labelids.txt"
        self.scannet_mapping = self.load_class_mapping(self.scannet_classes_path)
        assert len(self.scannet_mapping) == 20

        self.nyu40_to_scannet[list(self.scannet_mapping.keys())] = np.arange(
            len(self.scannet_mapping)
        )
        # scannet -> nyu40
        self.scannet_to_nyu40 = np.array(
            list(self.scannet_mapping.keys()) + [0], dtype=np.int64
        )

    def load_class_mapping(self, filename):
        id_to_class = OrderedDict()
        with open(filename, "r") as f:
            for line in f.readlines():
                class_id, class_name = line.rstrip().split("\t")
                id_to_class[int(class_id)] = class_name
        return id_to_class

    def list_labels(self):
        labels = {}
        for ctx in tqdm(self):
            # loading label
            _unique_lables = np.unique(ctx["label"])
            # increment count of each available label
            for _unique_label in _unique_lables:
                if int(_unique_label) != -100:
                    label_name = list(self.scannet_mapping.values())[_unique_label]
                    if label_name in labels:
                        labels[label_name] = labels[label_name] + 1

                    else:
                        labels[label_name] = 1

        label_table = PrettyTable()
        label_table.field_names = ["label", "count"]
        for name, count in labels.items():
            label_table.add_row([name, count])

        print(label_table)
        return labels

    def collect_from_label(self, label, max_to_keep: int = 10, skip: int = 4):
        label_names = list(self.scannet_mapping.values())
        if not label in label_names:
            logging.error("Label {} does not exist in {}".format(label, label_names))
            exit()
        # query index of label name
        target_label_index = label_names.index(label)
        # this will store all samples that include label
        collector = []

        logging.info("Start to query samples")
        for index in tqdm(range(0, len(self), skip)):
            sample_dict = self[index]
            label_indexes = np.unique(sample_dict["label"])
            for label_index in label_indexes:
                # avoid non-valid label (-100)
                if int(label_index) != -100:
                    if int(label_index) == target_label_index:
                        if len(collector) < max_to_keep:
                            collector.append(sample_dict)
                            continue

            if len(collector) == max_to_keep:
                break

        return collector

    def __getitem__(self, idx):
        color = self.colors[idx]
        depth = self.depths[idx]
        pose = self.poses[idx]
        label = self.labels[idx]
        scanid = Path(color).parents[1].name
        intrinsic = self.scannet_2d_dir / scanid / "intrinsic" / "intrinsic_depth.txt"
        check_path(color)
        check_path(depth)
        check_path(pose)
        check_path(label)
        check_path(intrinsic)

        color_tensor = np.array(Image.open(color))
        depth_tensor = Image.open(depth)
        depth_tensor = np.asarray(depth_tensor, dtype=np.float32) / self.__depth_scale
        pose_tensor = np.loadtxt(pose)
        label_tensor = np.array(Image.open(label))
        cam_matrix = np.loadtxt(intrinsic, dtype=np.float32)

        # overwrite bad labels
        bad_label_ind = np.logical_or(label_tensor < 0, label_tensor > 40)
        if bad_label_ind.any():
            label_tensor[bad_label_ind] = 0

        label_tensor = self.nyu40_to_scannet[label_tensor].astype(np.int64)

        out_dict = {
            "scanid": scanid,
            "color_path": color,
            "pose_path": pose,
            "depth_path": depth,
            "color": color_tensor,
            "depth": depth_tensor,
            "label": label_tensor,
            "pose": pose_tensor,
            "cam_matrix": cam_matrix,
        }

        return out_dict

    def __len__(self):
        return len(self.colors)


class Search3D:
    # related to ICP and RMSE calculation
    ICP_THRESHOLD = 0.5
    ICP_MAX_ITERATION = 10_000
    # related to KDTreeFlann search
    KNN_RADIUS = 0.5
    KNN_NN = 1

    def __init__(self, scans_dir: Path) -> None:
        self.scans_dir = Path(scans_dir)
        check_path(scans_dir)
        self.all_scenes = self.get_scenes(self.scans_dir)

    def read_ids(self, path):
        scannet_3d_dir = path
        check_path(path)
        dir_names = scannet_3d_dir.iterdir()
        dir_names = list(dir_names)  # convert to list from generator
        scan_ids = [name.name for name in dir_names]
        return scan_ids

    def get_scenes(self, path):
        scene_dir = path
        scan_ids = self.read_ids(path)
        return [scene_dir / scanid for scanid in scan_ids]

    def search(self, point_cloud, percentage=""):
        # store scenes and their distance to target
        all_knns = {}

        for idx, scene in enumerate(self.all_scenes):
            sample_id = scene.name
            sample_ply_scene_path = scene / "{}_vh_clean_2.ply".format(sample_id)
            check_path(sample_ply_scene_path)

            pc = read_pc_from_ply(sample_ply_scene_path, return_color=True)
            scene_point_cloud = draw_point_cloud(pc["points"])
            pcd_tree = o3d.geometry.KDTreeFlann(scene_point_cloud)
            overlaps = np.zeros([len(np.asarray(scene_point_cloud.points))], dtype=bool)

            for point_idx in range(len(point_cloud)):
                found, idx_point, dist = pcd_tree.search_hybrid_vector_3d(
                    point_cloud[point_idx, :3], self.KNN_RADIUS, self.KNN_NN
                )

                if found:
                    overlaps[idx_point] = True

            if True in np.unique(overlaps):
                # Define ICP parameters
                threshold = 0.5  # Maximum correspondence point-pair distance
                trans_init = np.identity(4)  # Initial transformation (identity matrix)
                max_iter = 5_000  # Maximum number of ICP iterations
                overlap_base_pts_vis = draw_point_cloud(
                    np.asarray(scene_point_cloud.points)[overlaps[:]]
                )

                point_cloud_obj = draw_point_cloud(point_cloud)
                reg_p2p = o3d.registration.registration_icp(
                    point_cloud_obj,
                    overlap_base_pts_vis,
                    self.ICP_THRESHOLD,
                    trans_init,
                    o3d.registration.TransformationEstimationPointToPoint(),
                    o3d.registration.ICPConvergenceCriteria(
                        max_iteration=self.ICP_MAX_ITERATION
                    ),
                )

                alignment_error = reg_p2p.inlier_rmse

                all_knns.update(
                    {alignment_error: {"overlap": overlaps, "id": sample_id}}
                )

        if len(all_knns) < 1:
            # no candidate found
            return "NC"

        query_distances = [dis for dis in all_knns.keys()]
        min_distance = min(query_distances)

        nearest_result = all_knns[min_distance]
        result_scan_id = nearest_result["id"]

        return result_scan_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        type=str,
        required=False,
        default=None,
        help="query on specific label",
    )
    parser.add_argument(
        "--keep",
        type=int,
        required=False,
        default=10,
        help="how many sample to keep",
    )
    parser.add_argument(
        "--skip",
        type=int,
        required=False,
        default=10,
        help="how many sample to skip in between",
    )
    parser.add_argument(
        "--no-mvpnet",
        action="store_true",
        required=False,
        help="no model segmentation",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        required=False,
        help="list available labels",
    )
    args = parser.parse_args()

    model_config = Path.cwd() / "configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml"
    check_path(model_config)

    output_directory = Path.cwd() / "outputs/scannet/mvpnet_3d_unet_resnet34_pn2ssg"
    check_path(output_directory)

    query_2d = Query2D(scannet_2d_dir=SCANNET_2D_DIRECTORY)

    if args.list:
        query_2d.list_labels()

    if not args.no_mvpnet:
        model = load_model(config=model_config)
        checkpointer = CheckpointerV2(model, save_dir=output_directory)
        if not checkpointer.has_checkpoint():
            logging.error("No checkpoint has been found")

        # this will resume and load last checkpoint
        checkpointer.load(None, resume=True)

    if args.label is not None:
        searcher = Search3D(scans_dir=SCANNET_DIRECTORY)
        confusion_labels = searcher.read_ids(path=Path(SCANNET_DIRECTORY))
        confusion_labels = [name.replace("scene", "") for name in confusion_labels]
        confusion_labels.insert(0, "NC")
        print(confusion_labels)

        results = query_2d.collect_from_label(
            label=args.label, max_to_keep=args.keep, skip=args.skip
        )

        true_scenes = []
        raw_predicted_scenes = []

        mvpnet_true_scenes = []
        mavpnet_predicted_scenes = []

        logging.info("Start 3D search")
        for ctx in tqdm(results):
            search_start_date = datetime.now()
            projection = unproject(ctx["cam_matrix"], ctx["depth"])
            projection = (
                ctx["pose"][:3, :3].dot(projection[:, :3].T).T + ctx["pose"][:3, 3]
            )

            searched_scanid = searcher.search(projection)
            target_scanid = ctx["scanid"]

            true_scenes.append(target_scanid.replace("scene", ""))
            raw_predicted_scenes.append(searched_scanid.replace("scene", ""))

            if not args.no_mvpnet:
                data = get_input_batch_sample(
                    points=projection,
                    color=Image.open(ctx["color_path"]),
                    depth=Image.open(ctx["depth_path"]),
                    pose=ctx["pose"],
                    cam_matrix=ctx["cam_matrix"],
                    depth_scale_factor=1000.0,
                )

                label_names = list(query_2d.scannet_mapping.values())
                if not args.label in label_names:
                    logging.error(
                        "Label {} does not exist in {}".format(args.label, label_names)
                    )
                    exit()

                choosen_label = label_names.index(args.label)
                segmentation = predict_segmentation(model=model, data_batch=data)
                scene_predicted_labels = np.argmax(segmentation, axis=1)

                label_mask = np.where(scene_predicted_labels == choosen_label)
                masked_unproj_pts = projection[label_mask]

                searched_scanid = searcher.search(masked_unproj_pts)
                target_scanid = ctx["scanid"]

                mvpnet_true_scenes.append(target_scanid.replace("scene", ""))
                mavpnet_predicted_scenes.append(searched_scanid.replace("scene", ""))

        cm_raw = calculate_confusion(
            true_scenes, raw_predicted_scenes, labels=confusion_labels
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_raw, display_labels=confusion_labels
        )

        disp.plot()
        plt.show()

        if not args.no_mvpnet:
            cm_mvpnet = calculate_confusion(
                mvpnet_true_scenes, mavpnet_predicted_scenes, labels=confusion_labels
            )

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm_mvpnet, display_labels=confusion_labels
            )

            disp.plot()
            plt.show()


if __name__ == "__main__":
    main()
