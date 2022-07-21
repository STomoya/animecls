
import os
import random
from typing import Callable

import numpy as np
from omegaconf import DictConfig
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from storch import get_now_string
from storch import hydra_utils
from storch.path import Path, Folder

def setup(config):
    '''Setup for the training loop'''
    folder = Folder(Path(config.config.run.ckpt_folder) / config.config.run.name / get_now_string())
    folder.mkdir()
    hydra_utils.save_hydra_config(config, folder.root / 'config.yaml')
    return folder


def count_correct(output: torch.Tensor, target: torch.Tensor, topk: int=1):
    '''Count correctly classified samples in topk.'''
    _, pred = output.topk(topk, dim=1)
    pred = pred.t()
    target = target.view(1, -1).expand_as(pred)
    correct = (pred == target).sum().item()
    return correct


def test_classification(
    targets: np.ndarray, predictions: np.ndarray, labels: list=None,
    filename: str=None, print_fn: Callable=print
):
    '''Calculate and visualizing scores for classification (mostly using sklearn).
    '''
    if labels is not None:
        labels = np.array(labels)
        targets = labels[targets]
        predictions = labels[predictions]

    print_fn('TEST')
    # precision, recall, F1, accuracy
    print_fn(
        f'Classification report:\n{classification_report(targets, predictions)}')

    # confusion matrix
    confmat = confusion_matrix(targets, predictions, labels=labels, normalize='true')
    print_fn(f'Confusion matrix:\n{confmat}')

    # visualize confmat
    fig, ax = plt.subplots(tight_layout=True)
    ax.matshow(confmat, cmap='Blues')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('prediction')
    ax.set_ylabel('ground truth')
    plt.savefig(filename)
    plt.close()


def _build_scheduler(config: DictConfig, optimizer: optim.Optimizer, iterations: int, total_iters: int):
    '''Build scheduler

    Arguments:
        config: omegaconf.DictConfig
            DictConfig object with 'name' and 'args' node.
            If name is multistep, also requires 'rel_milestone' node.
        optimizer: torch.optim.Optimizer
            The optimizer.
        iterations: int
            Number of iterations to apply the schedule.
        total_iters: int
            Total iterations in training.
    '''
    args = config.args if config.args is not None else {}
    if config.name == 'const':
        return lr_scheduler.ConstantLR(optimizer, total_iters=iterations, **args)
    if config.name == 'linear':
        return lr_scheduler.LinearLR(optimizer, total_iters=iterations, **args)
    if config.name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, **args)
    if config.name == 'step':
        return lr_scheduler.StepLR(optimizer, **args)
    if config.name == 'multistep':
        milestones = [int(total_iters * rel_milestone) for rel_milestone in config.rel_milestone]
        return lr_scheduler.MultiStepLR(optimizer, milestones, **args)


def build_scheduler(config: DictConfig, optimizer: optim.Optimizer, total_iterations: int):
    '''Buid scheduler from DictConfig

    Arguments:
        config: DictConfig
            Specifically config.config.train.optimizer node of the loaded config.
        optimizer: optim.Optimizer
            The optimizer
        total_iterations: int
            Total iterations in training.
    '''
    if config.name == 'chain':
        assert len(config.length) == len(config.schedulers)
        schedulers = []
        for index, parameters in enumerate(config.schedulers):
            iterations = total_iterations * config.length[index]
            schedulers.append(_build_scheduler(parameters, optimizer, iterations, total_iterations))
        return lr_scheduler.ChainedScheduler(schedulers)
    return _build_scheduler(config, optimizer, total_iterations, total_iterations)


def set_seeds(seed: int, use_deterministic_algorithms: bool=False, cudnn_benchmark: bool=True):
    '''Set seeds and optionally setup for reproducible training.

    Arguments:
        seed: int
            Seed value.
        use_deterministic_algorithms: bool (default: False)
            Use deterministic algorithm?
        cudnn_benchmark: bool (default: True)
            Set torch.backends.cudnn.benchmark with this value.
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = cudnn_benchmark
    if use_deterministic_algorithms:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(use_deterministic_algorithms, warn_only=True)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return seed_worker, generator
