from glob import glob
import os
from os import path
from pathlib import Path
from random import shuffle

from SETTING import ROOT_DIRECTORY, SCANNET_DIRECTORY

DATASET_DIRECTORY_3D = SCANNET_DIRECTORY
OUTPUT_META_DIR = os.path.join(ROOT_DIRECTORY, "meta")

#NOTE edit this
# all ratios should sum in 100
TRAIN_RATIO = 70
VALI_RATIO = 20
TEST_RATIO = 10

if (TRAIN_RATIO + VALI_RATIO + TEST_RATIO) != 100:
    print("[ERROR] ratio mismatch")
    exit()

if not Path(OUTPUT_META_DIR).exists():
    print("Creating folder {}".format(OUTPUT_META_DIR))
    os.mkdir(OUTPUT_META_DIR)

all_samples = glob(path.join(DATASET_DIRECTORY_3D, "*"))
all_samples = [Path(p).name for p in all_samples]
shuffle(all_samples)

def ratio(r):
    a = len(all_samples) / 100
    return round(r * a)
    
train_samples = all_samples[:ratio(TRAIN_RATIO)]
new_all_samples = all_samples[ratio(TRAIN_RATIO):]
validation_samples = new_all_samples[:ratio(VALI_RATIO)]
new_all_samples = new_all_samples[ratio(VALI_RATIO):]
test_samples = new_all_samples[:ratio(TEST_RATIO)]


train_meta = Path(OUTPUT_META_DIR) / "scannetv2_train.txt"
validation_meta = Path(OUTPUT_META_DIR) / "scannetv2_val.txt"
test_meta = Path(OUTPUT_META_DIR) / "scannetv2_test.txt"

def write(samples, path):
    print("[STATUS] writing to {}".format(path))
    with open(path, "a") as file:
        for s in samples:
            file.write(s + "\n")

write(train_samples, train_meta)
write(validation_samples, validation_meta)
write(test_samples, test_meta)