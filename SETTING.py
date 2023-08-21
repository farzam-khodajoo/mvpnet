# this will be to root of all dataset, which means 2D dataset, 
# metafiles and cached samples will be stored.
# after the process has been completed, this directory will contain a structure like this:
# - /content/dataset
# - - - resized_scans_120_160/
# - - - pickles/
# - - - metafiles/
ROOT_DIRECTORY = "/content/dataset"
# this directory is where you downloaded scannet dataset to.
# which should contain samples named in this format: scene0**_**
# this folder can also be inside ROOT_DIRECTORY, as it makes no difference.
SCANNET_DIRECTORY = "/content/dataset/scans"
# choose how many processing core you would like to be used.
MULTI_CORE_PROCESSING = 1