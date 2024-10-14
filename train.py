import random
from glob import glob

import numpy as np
import torch
from ignite.engine import Engine, Events
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from config import Config
from modules.dataset import SimulationDataset
from utils.load import (
    parse_multiple_sim_experiment_with_DVT,
    parse_sim_experiment_with_DVT,
)
from utils.pca import pca_torch

config = Config()


def process_function(engine, batch):
    X_batch, (y_spike_batch, y_soma_batch, y_DVT_batch) = batch
    return X_batch, y_spike_batch, y_soma_batch, y_DVT_batch


files = glob(config.nmda_dataset.data_dir + "*_6_secDuration_*")

valid_files = random.choice(files)

train_files = [f for f in files if f != valid_files]

data_dict = {"train_files": train_files, "valid_files": valid_files}


valid_files_per_epoch = max(
    1, int(config.training.validation_fraction * config.training.train_files_per_epoch)
)
dataset = SimulationDataset(
    train_files,
    valid_files_per_epoch,
    config.training.batch_size,
    config.training.window_size_ms,
    config.training.train_file_load,
    config.training.ignore_time_from_start,
    config.training.y_train_soma_bias,
    config.training.y_soma_threshold,
    config.training.y_DTV_threshold,
)
dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)

trainer = Engine(process_function)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_epoch(engine):
    dataset.shuffle()


# トレーニング開始
trainer.run(dataloader, max_epochs=5)
