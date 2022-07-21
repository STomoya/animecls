
from animecls.dataset import build_dataset
import storch
from animecls.train_loop.utils import build_scheduler
from storch.hydra_utils import get_hydra_config
from omegaconf import OmegaConf
config = get_hydra_config('config', 'config.yaml')

# print(OmegaConf.to_yaml(config))
# train, val, test = build_dataset(config.config.data)
# print(train.dataset.label2name)
# label2name = test.dataset.label2name
# label2name = sorted(label2name.items(), key=lambda x:x[0])
# names = [name for _, name in label2name]
# print(len(names))


cfg = config.config
tcfg = cfg.train
# model
model = storch.construct_class_by_name(**cfg.model)
# optimizer
optimizer = storch.construct_class_by_name(model.parameters(), **tcfg.optimizer)
scheduler = build_scheduler(tcfg.scheduler, optimizer, 10000)

lrs = []
for _ in range(10000):
    lrs.append(scheduler.get_last_lr())
    scheduler.step()
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.plot(range(len(lrs)), lrs)
plt.savefig('loss')
