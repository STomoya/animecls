'''Utilities for datasets'''

import glob
import os
import pickle

from omegaconf import DictConfig

import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from storch import EasyDict, construct_class_by_name
from storch.dataset.utils import is_image_file

def build_transform(config: DictConfig):
    '''build transform from list of arguments

    Arguments:
        config: omegaconf.DictConfig
            Specifically config.config.train.transforms.{train.test} node of the loaded config.
    '''
    transforms = []
    for arguments in config:
        if arguments.class_name == 'OneOf':
            inner_transforms = []
            for inner_args in arguments.transforms:
                inner_transforms.append(construct_class_by_name(**inner_args))
            transforms.append(T.RandomChoice(inner_transforms))
        transforms.append(construct_class_by_name(**arguments))

    return T.Compose(transforms)


def split_data(data_root: str):
    '''Split files inside data_root

    Arguments:
        data_root: str
            Path to the images.
    '''
    files = glob.glob(os.path.join(data_root, '*', '*'))
    files = [file for file in files if os.path.isfile(file) and is_image_file(file)]
    folders = [file.split('/')[-2] for file in files]

    train_files, valtest_files, train_folders, valtest_folders = train_test_split(files, folders, test_size=0.2)
    val_files, test_files, val_folders, test_folders = train_test_split(valtest_files, valtest_folders, test_size=0.5)

    splits = EasyDict()
    splits.train = EasyDict()
    splits.train.images = train_files
    splits.train.targets = train_folders
    splits.val = EasyDict()
    splits.val.images = val_files
    splits.val.targets = val_folders
    splits.test = EasyDict()
    splits.test.images = test_files
    splits.test.targets = test_folders

    return splits


def prepare_labels(folder_list, meta_root):
    '''Prepare labels and load meta data.

    Arguments:
        folder_list: list[str]
            List of folders representing the class.
        meta_root: str
            Path to the folder which contains 'folder2label.pickle' and 'label2name.pickle'.
    '''
    with open(os.path.join(meta_root, 'folder2label.pickle'), 'rb') as fin:
        folder2label = pickle.load(fin)
    with open(os.path.join(meta_root, 'label2name.pickle'), 'rb') as fin:
        label2name = pickle.load(fin)
    labels = [folder2label[folder] for folder in folder_list]
    return labels, label2name
