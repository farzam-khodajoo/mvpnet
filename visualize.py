import yaml
from pathlib import Path
from mvpnet.config.mvpnet_3d import cfg
from mvpnet.models.build import build_model_mvpnet_3d
from mvpnet.data.build import build_dataloader

from mvpnet.utils.o3d_util import draw_point_cloud
from mvpnet.utils.visualize import label2color
import open3d as o3d

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


cfg.merge_from_file("/content/mvpnet/configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml")
model, loss_fn, train_metric, val_metric = build_model_mvpnet_3d(cfg)
model = model.cuda()
train_dataloader = build_dataloader(cfg, mode='train')
batch = next(iter(train_dataloader))
batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
preds = model(batch)

print("size of point clouds: ", batch["points"].size())
print("size of segmentation logit: ", preds["seg_logit"].size())

batch = {k: v.cpu() for k, v in batch.items()}
points = np.asarray(batch["points"])

views = [(90, 90), (90, -90), (0, 180), (180, 0)]
for (x, y) in views:
    ax = plt.axes(projection='3d')
    ax.view_init(x, y)
    ax.axis("off")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
    print("Saving {} ..".format(Path.cwd() / Path("view_mvpnet_{}_{}.png".format(x, y))))
    plt.savefig(Path.cwd() / Path("view_mvpnet_{}_{}.png".format(x, y)))