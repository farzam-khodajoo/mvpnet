import yaml
from mvpnet.config.mvpnet_3d import cfg
from mvpnet.models.build import build_model_mvpnet_3d
from mvpnet.data.build import build_dataloader

from mvpnet.utils.o3d_util import draw_point_cloud
from mvpnet.utils.visualize import label2color
import open3d as o3d

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import sys
import os
is_windows = sys.platform == "win32"

if is_windows:
    path = "{}/configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml".format(os.getcwd())
    path = path.replace("/", "\\")
    print(path)
    cfg.merge_from_file(path)
else:
    cfg.merge_from_file("/content/mvpnet/configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml")


model, loss_fn, train_metric, val_metric = build_model_mvpnet_3d(cfg)
model = model.cuda()
train_dataloader = build_dataloader(cfg, mode='train')
batch = next(iter(train_dataloader))
batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
preds = model(batch)

print(batch["points"].size())
print(preds["seg_logit"].size())


#colors = label2color(preds["seg_logit"][0])

batch = {k: v.cpu() for k, v in batch.items()}
points = np.asarray(batch["points"])

ax = plt.axes(projection='3d')
ax.view_init(90, 90)
ax.axis("off")
ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
plt.savefig("result.png")