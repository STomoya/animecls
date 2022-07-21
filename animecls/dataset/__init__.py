'''Datasets'''

from __future__ import annotations

from typing import Callable
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from animecls.dataset.dataset import ImageFolder
from animecls.dataset.utils import split_data, build_transform


def build_dataset(
    config: DictConfig, worker_init_fn: Callable=None, generator: torch.Generator=None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    '''build dataset

    Arguments:
        config: omegaconf.DictConfig
            Configuration obj from omegaconf. Specifically config.config.data node of the loaded configuration.
        worker_init_fn: Callable
            Function to initialize all workers for DataLoader.
            Used only for the training set which only sets shuffle=True.
        generator: torch.Generator
            torch.Generator object.
            Used only for the training set which only sets shuffle=True.
    '''
    data_splits = split_data(config.data_root)

    train_transform = build_transform(config.transforms.train)
    test_transform = build_transform(config.transforms.test)

    train_split = data_splits.train
    train_dataset = ImageFolder(train_split.images, train_split.targets, config.meta_root, train_transform)
    train_loader = DataLoader(train_dataset, **config.loader, worker_init_fn=worker_init_fn, generator=generator)

    val_split = data_splits.val
    val_dataset = ImageFolder(val_split.images, val_split.targets, config.meta_root, test_transform)
    val_loader = DataLoader(val_dataset, config.loader.batch_size, shuffle=False)

    test_split = data_splits.test
    test_dataset = ImageFolder(test_split.images, test_split.targets, config.meta_root, test_transform)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)

    return train_loader, val_loader, test_loader
