import os
import os.path as osp
import sys
import logging
import time
import warnings
import csv
from pathlib import Path
from collections import OrderedDict

extend_path = os.getcwd()
if sys.platform == "win32":
    extend_path = extend_path.replace("/", "\\")

sys.path.append(extend_path)

import numpy as np
from PIL import Image
from SETTING import ROOT_DIRECTORY, SCANNET_DIRECTORY

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import functional as F

from common.utils.checkpoint import CheckpointerV2
from common.utils.metric_logger import MetricLogger
from common.utils.torch_util import set_random_seed
from mvpnet.data.scannet_2d3d import ScanNet2D3DChunksTest
from mvpnet.evaluate_3d import Evaluator

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

def load_label_dictionary():

    # create label mapping
    all_class_ids = Path.cwd() / "mvpnet/data/meta_files/labelids.txt"
    label_mapping = load_class_mapping(all_class_ids)

    return label_mapping

def get_available_labels(labels_in_scene, label_dict):
    label_names = []
    for label_id in labels_in_scene:
        for idx, (_, name) in enumerate(label_dict.items()):
            if idx == label_id:
                label_names.append("{} [{}]".format(name, idx))

    return label_names


def inference_mvpnet(target_scene_id):
    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from mvpnet.models.build import build_model_mvpnet_3d
    from common.config import purge_cfg
    from mvpnet.config.mvpnet_3d import cfg
    config_file = Path.cwd() / "configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml"
    logging.info("Config file {} -> exists: {}".format(config_file, config_file.exists()))
    cfg.merge_from_file(str(config_file))
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs', 'outputs'))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)


    logger = logging.getLogger('mvpnet.test')

    # build mvpnet model
    model = build_model_mvpnet_3d(cfg)[0]
    model = model.cuda()

    # build checkpointer
    checkpointer = CheckpointerV2(model, save_dir=output_dir, logger=logger)
    # load last checkpoint
    checkpointer.load(None, resume=True)

    # build dataset
    k = 3
    num_views = 5
    min_nb_pts = 2048
    test_dataset = ScanNet2D3DChunksTest(cache_dir=str(Path(ROOT_DIRECTORY)/"pickles"),
                                         image_dir=str(Path(ROOT_DIRECTORY)/"scans_resize_160x120"),
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
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=lambda x: x[0])

    class_names = test_dataset.class_names
    num_classes = len(class_names)

    model.eval()
    set_random_seed(cfg.RNG_SEED)

    with torch.no_grad():
        for scan_idx, data_dict_list in enumerate(test_dataloader):
            scan_id = test_dataset.scan_ids[scan_idx]
            if not scan_id == target_scene_id:
                continue

            points = test_dataset.data[scan_idx]['points'].astype(np.float32)

            # prepare outputs
            num_points = len(points)
            pred_logit_whole_scene = np.zeros([num_points, num_classes], dtype=np.float32)
            num_pred_per_point = np.zeros(num_points, dtype=np.uint8)

            # iterate over chunks
            for data_dict in data_dict_list:
                chunk_ind = data_dict.pop('chunk_ind')

                # padding for chunks with points less than min-nb-pts
                # It is required since farthest point sampling requires more points than centroids.
                chunk_points = data_dict['points']
                chunk_nb_pts = chunk_points.shape[1]  # note that already transposed
                if chunk_nb_pts < min_nb_pts:
                    print('Too sparse chunk in {} with {} points.'.format(scan_id, chunk_nb_pts))
                    pad = np.random.randint(chunk_nb_pts, size=min_nb_pts - chunk_nb_pts)
                    choice = np.hstack([np.arange(chunk_nb_pts), pad])
                    data_dict['points'] = data_dict['points'][:, choice]
                    data_dict['knn_indices'] = data_dict['knn_indices'][choice]

                data_batch = {k: torch.tensor([v]) for k, v in data_dict.items()}
                data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}
                # forward 
                preds = model(data_batch)
                seg_logit = preds['seg_logit'].squeeze(0).cpu().numpy().T
                seg_logit = seg_logit[:len(chunk_ind)]
                # update
                pred_logit_whole_scene[chunk_ind] += seg_logit
                num_pred_per_point[chunk_ind] += 1

            pred_logit_whole_scene = pred_logit_whole_scene / np.maximum(num_pred_per_point[:, np.newaxis], 1)
            pred_label_whole_scene = np.argmax(pred_logit_whole_scene, axis=1)

            no_pred_mask = num_pred_per_point == 0
            no_pred_indices = np.nonzero(no_pred_mask)[0]
            if no_pred_indices.size > 0:
                logger.warning('{:s}: There are {:d} points without prediction.'.format(scan_id, no_pred_mask.sum()))
                pred_label_whole_scene[no_pred_indices] = num_classes

            return points, pred_label_whole_scene

def inference_2d_chunks(target_scene_id):
    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from mvpnet.models.build import build_model_sem_seg_2d
    from mvpnet.models.mvpnet_2d import MVPNet2D

    from common.config import purge_cfg
    from mvpnet.config.sem_seg_2d import cfg
    config_file = Path.cwd() / "configs/scannet/unet_resnet34.yaml"
    logging.info("Config file {} -> exists: {}".format(config_file, config_file.exists()))
    cfg.merge_from_file(str(config_file))
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs', 'outputs'))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    # build 2d model
    net_2d = build_model_sem_seg_2d(cfg)[0]

    # build checkpointer
    logger = logging.getLogger('mvpnet.test')
    checkpointer = CheckpointerV2(net_2d, save_dir=output_dir, logger=logger)
 
    checkpointer.load(None, resume=True)

    # wrapper for 2d model
    model = MVPNet2D(net_2d)
    model = model.cuda()

    # build dataset
    k = 1
    num_views = 5
    min_nb_pts = 2048
    test_dataset = ScanNet2D3DChunksTest(cache_dir=str(Path(ROOT_DIRECTORY)/"pickles"),
                                         image_dir=str(Path(ROOT_DIRECTORY)/"scans_resize_160x120"),
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
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=lambda x: x[0])

    class_names = test_dataset.class_names
    num_classes = len(class_names)

    model.eval()
    set_random_seed(cfg.RNG_SEED)

    with torch.no_grad():
        for scan_idx, data_dict_list in enumerate(test_dataloader):
            scan_id = test_dataset.scan_ids[scan_idx]
            if not scan_id == target_scene_id:
                continue

            # fetch data
            scan_id = test_dataset.scan_ids[scan_idx]
            points = test_dataset.data[scan_idx]['points'].astype(np.float32)
            seg_label = test_dataset.data[scan_idx]['seg_label']
            seg_label = test_dataset.nyu40_to_scannet[seg_label]

            # prepare outputs
            num_points = len(points)
            pred_logit_whole_scene = np.zeros([num_points, num_classes], dtype=np.float32)
            num_pred_per_point = np.zeros(num_points, dtype=np.uint8)

            # iterate over chunks
            for data_dict in data_dict_list:
                chunk_ind = data_dict.pop('chunk_ind')
                data_batch = {k: torch.tensor([v]) for k, v in data_dict.items()}
                data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}
                # forward
                preds = model(data_batch)
                seg_logit = preds['seg_logit'].squeeze(0).cpu().numpy().T
                seg_logit = seg_logit[:len(chunk_ind)]
                # update
                pred_logit_whole_scene[chunk_ind] += seg_logit
                num_pred_per_point[chunk_ind] += 1

            pred_logit_whole_scene = pred_logit_whole_scene / np.maximum(num_pred_per_point[:, np.newaxis], 1)
            pred_label_whole_scene = np.argmax(pred_logit_whole_scene, axis=1)

            no_pred_mask = num_pred_per_point == 0
            no_pred_indices = np.nonzero(no_pred_mask)[0]
            if no_pred_indices.size > 0:
                logger.warning('{:s}: There are {:d} points without prediction.'.format(scan_id, no_pred_mask.sum()))
                pred_label_whole_scene[no_pred_indices] = num_classes


            return points, pred_label_whole_scene
        
def inference_2d(image_path):
    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from mvpnet.models.build import build_model_mvpnet_3d
    from common.config import purge_cfg
    from mvpnet.config.mvpnet_3d import cfg
    config_file = Path.cwd() / "configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml"
    logging.info("Config file {} -> exists: {}".format(config_file, config_file.exists()))
    cfg.merge_from_file(str(config_file))
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs', 'outputs'))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)


    logger = logging.getLogger('mvpnet.test')

    # build mvpnet model
    model = build_model_mvpnet_3d(cfg)[0]
    model = model.cuda()

    # build checkpointer
    checkpointer = CheckpointerV2(model, save_dir=output_dir, logger=logger)
    # load last checkpoint
    checkpointer.load(None, resume=True)

    model.eval()
    set_random_seed(cfg.RNG_SEED)

    import torch
    from torch import nn
    import numpy as np

    from common.nn import SharedMLP
    from mvpnet.ops.group_points import group_points
    from PIL import Image

    image = Image.open(image_path)
    image = np.array(image.resize((160, 120), Image.BILINEAR))
    image = np.asarray(image, dtype=np.float32) / 255.0

    images = np.stack([image], axis=0)  # (nv, h, w, 3)
    print("images => ", images.shape)
    data_batch = {"images": images.astype(np.float32, copy=False)}
    data_batch["images"] = np.moveaxis(
            data_batch["images"], -1, 1
        )  # (nv, 3, h, w)

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
    feature_2d = preds_2d['feature']  # (b * nv, c, h, w
    resnet_output = feature_2d.cpu().squeeze(0).detach().numpy().T
    resnet_output = np.argmax(resnet_output, axis=2)

    return resnet_output