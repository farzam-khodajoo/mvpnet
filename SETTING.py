# state switches between training data path and test data path
# options: "train" or "test"
STATE = "test"

# dataset path for training
TRAIN_ROOT_DIRECTORY = "C:\\Users\\Arsham\\Desktop\\Projects\\mvpnet-all\\dataset\\scannet"
TRAIN_SCANNET_DIRECTORY = "C:\\Users\\Arsham\\Desktop\\Projects\\mvpnet-all\\dataset\\scannet\\scans"

# dataset path for test (or inference)
TEST_ROOT_DIRECTORY = "C:\\Users\\Arsham\\Desktop\\Projects\\mvpnet-all\\dataset\\scannet"
TEST_SCANNET_DIRECTORY = "C:\\Users\\Arsham\\Desktop\\Projects\\mvpnet-all\\dataset\\scannet\\scans"

# this will be to root of all dataset, which means 2D dataset, 
# metafiles and cached samples will be stored.
# after the process has been completed, this directory will contain a structure like this:
# - /content/dataset
# - - - resized_scans_120_160/
# - - - pickles/
# - - - metafiles/
ROOT_DIRECTORY = TRAIN_ROOT_DIRECTORY if STATE == "train" else TEST_ROOT_DIRECTORY
# this directory is where you downloaded scannet dataset to.
# which should contain samples named in this format: scene0**_**
# this folder can also be inside ROOT_DIRECTORY, as it makes no difference.
SCANNET_DIRECTORY = TRAIN_SCANNET_DIRECTORY if STATE == "train" else TEST_SCANNET_DIRECTORY
# choose how many processing core you would like to be used.
MULTI_CORE_PROCESSING = 1