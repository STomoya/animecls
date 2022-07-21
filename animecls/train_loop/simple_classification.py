
import copy
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler

import storch
from storch.torchops import auto_get_device, optimizer_step, freeze
from storch.status import Status
from storch.metrics import test_classification

from animecls.dataset import build_dataset
from animecls.train_loop import utils

def train(config):
    folder = utils.setup(config)
    cfg = config.config
    tcfg = cfg.train

    worker_init_fn, generator = utils.set_seeds(cfg.env.random_seed, cfg.env.use_deterministic_algorithm, cfg.env.cudnn_benchmark)

    device = auto_get_device()
    amp = cfg.env.amp
    scaler = GradScaler() if amp else None

    # dataset
    train_data, val_data, test_data = build_dataset(cfg.data, worker_init_fn, generator)

    # model
    model = storch.construct_class_by_name(**cfg.model.parameters)
    model.to(device)

    # optimizer
    optimizer = storch.construct_class_by_name(model.parameters(), **tcfg.optimizer)
    scheduler = utils.build_scheduler(tcfg.scheduler, optimizer, len(train_data)*tcfg.epochs)

    # criterion
    criterion = storch.construct_class_by_name(**tcfg.criterion)

    # status
    status = Status(len(train_data)*tcfg.epochs, False, folder.root / cfg.run.log_file, cfg.run.log_interval, cfg.run.name)
    status.initialize_collector(
        'Loss/CE/train', 'Loss/CE/val', 'Acc@top1/train', 'Acc@top1/val', 'Acc@top5/train', 'Acc@top5/val')
    status.log_stuff(cfg, model, optimizer, train_data)

    best_loss = 1e10

    epochs = 0
    while not status.is_end():
        epochs += 1
        model.train()
        for input, target in train_data:
            input = input.to(device)
            target = target.to(device)

            with autocast(amp):
                output = model(input)
                batch_loss = criterion(output, target)

            optimizer_step(batch_loss, optimizer, scaler, zero_grad=True, set_to_none=True, update_scaler=True)
            scheduler.step()

            correct_top1 = utils.count_correct(output, target, topk=1)
            correct_top5 = utils.count_correct(output, target, topk=5)

            status.update(**{
                'Loss/CE/train': batch_loss.item(),
                'Acc@top1/train': correct_top1 / target.size(0),
                'Acc@top5/train': correct_top5 / target.size(0)
            })

        model.eval()
        loss = 0
        correct_top1 = 0
        correct_top5 = 0
        with torch.no_grad(), status.stop_timer(verbose=True):
            for input, target in val_data:
                input = input.to(device)
                target = target.to(device)

                with autocast(amp):
                    output = model(input)
                    batch_loss = criterion(output, target)

                loss += batch_loss.item() * target.size(0)
                correct_top1 += utils.count_correct(output, target, topk=1)
                correct_top5 += utils.count_correct(output, target, topk=5)

        loss = loss / len(val_data.dataset)
        accuracy_top1 = correct_top1 / len(val_data.dataset)
        accuracy_top5 = correct_top5 / len(val_data.dataset)

        status.update_collector(**{
            'Loss/CE/val': loss,
            'Acc@top1/val': accuracy_top1,
            'Acc@top5/val': accuracy_top5})
        status.log(f'[VALIDATION] Loss: {loss}, Accuracy: (@top1) {accuracy_top1} (@top5) {accuracy_top5}')

        if loss < best_loss:
            best_loss = loss
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, folder.root / 'best_model.torch')

    status.plot(folder.root / 'status')

    model.load_state_dict(best_state_dict)
    freeze(model)
    preds, targets = [], []
    loss, correct_top1, correct_top5 = 0, 0, 0
    with torch.no_grad():
        for input, target in test_data:
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            batch_loss = criterion(output, target)

            loss += batch_loss.item() * target.size(0)
            correct_top1 += utils.count_correct(output, target, topk=1)
            correct_top5 += utils.count_correct(output, target, topk=5)
            _, pred = output.max(1)
            preds.append(pred.item())
            targets.append(target.item())

    loss = loss / len(test_data.dataset)
    accuracy_top1 = correct_top1 / len(val_data.dataset)
    accuracy_top5 = correct_top5 / len(val_data.dataset)
    status.log(f'[TEST] Loss: {loss}, Accuracy: (@top1) {accuracy_top1} (@top5) {accuracy_top5}')
    label2name = test_data.dataset.label2name
    label2name = sorted(label2name.items(), key=lambda x:x[0])
    names = [name for _, name in label2name]
    utils.test_classification(np.array(targets), np.array(preds), names, folder.root / 'cm', status.log)
