import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from config import Config
from modules.csnt import CSnT
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
train_dataset = SimulationDataset(
    train_files,
    valid_files_per_epoch,
    config.training.batch_size,
    config.training.window_size_ms,
    config.training.train_file_load,
    config.training.ignore_time_from_start,
    config.training.y_train_soma_bias,
    config.training.v_threshold,
    config.training.y_DTV_threshold,
    config.training.curr_file_index,
)
val_dataset = SimulationDataset(
    valid_files,
    valid_files_per_epoch,
    config.training.batch_size,
    config.training.window_size_ms,
    config.training.train_file_load,
    config.training.ignore_time_from_start,
    config.training.y_train_soma_bias,
    config.training.v_threshold,
    config.training.y_DTV_threshold,
    config.training.curr_file_index,
    is_shuffle=False,
)
train_dataloader = DataLoader(
    train_dataset, batch_size=config.training.batch_size, shuffle=False
)
val_dataloader = DataLoader(
    val_dataset, batch_size=config.training.batch_size, shuffle=False
)


input_size = train_dataset[0][0].shape[-1]
hidden_size = config.model.hidden_size
output_size = config.model.output_size
num_layers = config.model.num_layers
nhead = config.model.nhead
dropout = config.model.dropout

model = CSnT(input_size, hidden_size, output_size, num_layers, nhead, dropout)


criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()

    X_batch, (y_spike_batch, y_soma_batch, y_DVT_batch) = batch

    outputs = model(X_batch)
    y_spike_pred, y_soma_pred, y_DVT_pred = outputs.chunk(3, dim=-1)

    loss_spike = criterion(y_spike_pred, y_spike_batch)
    loss_soma = criterion(y_soma_pred, y_soma_batch)
    loss_DVT = criterion(y_DVT_pred, y_DVT_batch)

    total_loss = loss_spike + loss_soma + loss_DVT + model.l1_regularization()
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return total_loss.item()


def eval_step(engine, batch):
    model.eval()
    with torch.no_grad():
        X_batch, (y_spike_batch, y_soma_batch, y_DVT_batch) = batch
        outputs = model(X_batch)
        y_spike_pred, y_soma_pred, y_DVT_pred = outputs.chunk(3, dim=-1)

        loss_spike = criterion(y_spike_pred, y_spike_batch)
        loss_soma = criterion(y_soma_pred, y_soma_batch)
        loss_DVT = criterion(y_DVT_pred, y_DVT_batch)

        total_loss = loss_spike + loss_soma + loss_DVT

    return total_loss.item()


trainer = Engine(train_step)
evaluator = Engine(eval_step)

train_loss_metric = Loss(criterion)
train_loss_metric.attach(trainer, "train_loss")

val_loss_metric = Loss(criterion)
val_loss_metric.attach(evaluator, "val_loss")


@trainer.on(Events.EPOCH_COMPLETED)
def reset_epoch(engine):
    train_dataset.shuffle()


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    print(
        f"Epoch {engine.state.epoch}, Train Loss: {engine.state.metrics['train_loss']:.4f}"
    )


@evaluator.on(Events.COMPLETED)
def log_validation_results(engine):
    print(f"Validation Loss: {engine.state.metrics['val_loss']:.4f}")


@trainer.on(Events.EPOCH_COMPLETED)
def run_validation(engine):
    evaluator.run(val_dataloader)


early_stopping_handler = EarlyStopping(
    patience=10,
    score_function=lambda engine: -engine.state.metrics["val_loss"],
    trainer=trainer,
)
evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

checkpoint_handler = ModelCheckpoint(
    "models",
    "snn_transformer",
    save_interval=1,
    n_saved=3,
    create_dir=True,
    score_function=lambda engine: -engine.state.metrics["val_loss"],
    score_name="val_loss",
)
evaluator.add_event_handler(
    Events.COMPLETED, checkpoint_handler, to_save={"model": model}
)

trainer.run(train_dataloader, max_epochs=100)
scheduler.step()
