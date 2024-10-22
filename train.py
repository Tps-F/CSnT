import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, ProgressBar
from ignite.handlers.param_scheduler import LRScheduler
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
from utils.utils import get_experiment_dir

config = Config()


def process_function(engine, batch):
    X_batch, (y_spike_batch, y_soma_batch, y_DVT_batch) = batch
    return X_batch, y_spike_batch, y_soma_batch, y_DVT_batch


files = glob(config.nmda_dataset.data_dir + "*_6_secDuration_*")

valid_files = [random.choice(files)]

train_files = [f for f in files if f != valid_files]

data_dict = {"train_files": train_files, "valid_files": valid_files}


valid_files_per_epoch = max(
    1, int(config.training.validation_fraction * config.training.train_files_per_epoch)
)
train_dataset = SimulationDataset(
    train_files,
    valid_files_per_epoch,
    config.training.window_size_ms,
    config.training.train_file_load,
    config.training.ignore_time_from_start,
    config.training.y_train_soma_bias,
    config.training.v_threshold,
    config.training.y_DTV_threshold,
    config.training.curr_file_index,
    is_shuffle=True,
)
val_dataset = SimulationDataset(
    valid_files,
    len(valid_files),
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
output_sizes = config.model.output_sizes
num_layers = config.model.num_layers
nhead = config.model.nhead
dropout = config.model.dropout

model = CSnT(input_size, hidden_size, output_sizes, num_layers, nhead, dropout).to(
    config.device
)


criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)


scheduler = LRScheduler(
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
)


def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()

    X_batch, (y_spike_batch, y_soma_batch, y_DVT_batch) = batch
    X_batch = X_batch.to(config.device)
    y_spike_batch = y_spike_batch.to(config.device)
    y_soma_batch = y_soma_batch.to(config.device)
    y_DVT_batch = y_DVT_batch.to(config.device)

    y_spike_pred, y_soma_pred, y_DVT_pred = model(X_batch)

    loss_spike = criterion(y_spike_pred, y_spike_batch)
    loss_soma = criterion(y_soma_pred, y_soma_batch)
    loss_DVT = criterion(y_DVT_pred, y_DVT_batch)

    total_loss = (
        loss_spike * config.learning_rate_per_epoch[0]
        + loss_soma * config.learning_rate_per_epoch[1]
        + loss_DVT * config.learning_rate_per_epoch[2]
        + model.l1_regularization()
    )
    total_loss.backward(retain_graph=True)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return {
        "y_pred": (y_spike_pred, y_soma_pred, y_DVT_pred),
        "y": (y_spike_batch, y_soma_batch, y_DVT_batch),
        "loss": total_loss.item(),
        "loss_spike": loss_spike.item(),
        "loss_soma": loss_soma.item(),
        "loss_DVT": loss_DVT.item(),
    }


def eval_step(engine, batch):
    model.eval()
    with torch.no_grad():
        X_batch, (y_spike_batch, y_soma_batch, y_DVT_batch) = batch
        X_batch = X_batch.to(config.device)
        y_spike_batch = y_spike_batch.to(config.device)
        y_soma_batch = y_soma_batch.to(config.device)
        y_DVT_batch = y_DVT_batch.to(config.device)

        y_spike_pred, y_soma_pred, y_DVT_pred = model(X_batch)

        loss_spike = criterion(y_spike_pred, y_spike_batch)
        loss_soma = criterion(y_soma_pred, y_soma_batch)
        loss_DVT = criterion(y_DVT_pred, y_DVT_batch)

        total_loss = loss_spike + loss_soma + loss_DVT

    return {
        "y_pred": (y_spike_pred, y_soma_pred, y_DVT_pred),
        "y": (y_spike_batch, y_soma_batch, y_DVT_batch),
        "loss": total_loss.item(),
        "loss_spike": loss_spike.item(),
        "loss_soma": loss_soma.item(),
        "loss_DVT": loss_DVT.item(),
    }


trainer = Engine(train_step)
evaluator = Engine(eval_step)


Loss(criterion, output_transform=lambda x: (x["y_pred"], x["y"])).attach(
    trainer, "loss"
)
Loss(criterion, output_transform=lambda x: (x["y_pred"], x["y"])).attach(
    evaluator, "loss"
)


ProgressBar().attach(trainer)
ProgressBar().attach(evaluator)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_epoch(engine):
    train_dataset.shuffle()


@trainer.on(Events.EPOCH_COMPLETED)
def run_validation(engine):
    evaluator.run(val_dataloader)
    metrics = evaluator.state.metrics
    print(f"Epoch {engine.state.epoch}, Val Loss: {metrics['loss']:.4f}")


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    metrics = engine.state.output
    print(f"Iteration {engine.state.iteration}")
    print(f"Spike Loss: {metrics['loss_spike']:.4f}")
    print(f"Soma Loss: {metrics['loss_soma']:.4f}")
    print(f"DVT Loss: {metrics['loss_DVT']:.4f}")


@trainer.on(Events.ITERATION_COMPLETED)
def log_grad_norm(engine):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    print(f"Gradient norm: {total_norm:.4f}")


early_stopping_handler = EarlyStopping(
    patience=10,
    score_function=lambda engine: -engine.state.metrics["loss"],
    trainer=trainer,
)
evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

checkpoint_handler = ModelCheckpoint(
    get_experiment_dir(),
    "snn_transformer",
    n_saved=3,
    create_dir=True,
    score_function=lambda engine: -engine.state.metrics["loss"],
    score_name="loss",
)
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

evaluator.add_event_handler(
    Events.COMPLETED, checkpoint_handler, to_save={"model": model}
)

trainer.run(train_dataloader, max_epochs=config.num_epochs)
