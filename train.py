import os
import gc
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, ProgressBar
from ignite.metrics import RunningAverage
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config import Config
from modules.c2 import CSnT
from modules.dataset import SimulationDataset
from utils.utils import get_experiment_dir

# CUDA memory optimization settings
torch.cuda.empty_cache()
gc.collect()
torch.backends.cudnn.benchmark = True

config = Config()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
experiment_dir = get_experiment_dir()
writer = SummaryWriter(log_dir=experiment_dir + "tensorboard_logs")

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.spike_criterion = nn.BCEWithLogitsLoss()
        self.regression_criterion = nn.MSELoss()

    def forward(self, predictions, targets, epoch_weights):
        spike_pred, soma_pred, dvt_pred = predictions
        spike_true, soma_true, dvt_true = targets

        spike_loss = self.spike_criterion(spike_pred, spike_true)
        soma_loss = self.regression_criterion(soma_pred, soma_true)
        dvt_loss = self.regression_criterion(dvt_pred, dvt_true)

        temporal_consistency = (
            torch.mean(torch.abs(torch.diff(soma_pred, dim=1))) +
            torch.mean(torch.abs(torch.diff(dvt_pred, dim=1)))
        )

        total_loss = (
            spike_loss * epoch_weights[0] +
            soma_loss * epoch_weights[1] +
            dvt_loss * epoch_weights[2] +
            0.1 * temporal_consistency
        )

        return total_loss, {
            "spike_loss": spike_loss.item(),
            "soma_loss": soma_loss.item(),
            "dvt_loss": dvt_loss.item(),
            "temporal_loss": temporal_consistency.item(),
        }

@torch.no_grad()
def compute_metrics(y_pred, y_true):
    spike_pred, soma_pred, dvt_pred = y_pred
    spike_true, soma_true, dvt_true = y_true

    # Compute metrics on CPU to save GPU memory
    spike_pred_flat = spike_pred.sigmoid().cpu().numpy().ravel()
    spike_true_flat = spike_true.cpu().numpy().ravel()

    try:
        unique_classes = np.unique(spike_true_flat)
        spike_auc = (np.mean(spike_pred_flat) if unique_classes[0] == 1 else 1.0 - np.mean(spike_pred_flat)) if len(unique_classes) < 2 else roc_auc_score(spike_true_flat, spike_pred_flat)
    except Exception:
        spike_auc = 0.5

    soma_var = 1 - ((soma_pred - soma_true).pow(2).mean() / soma_true.var())
    dvt_corr = torch.corrcoef(torch.stack([dvt_pred.flatten(), dvt_true.flatten()]))[0, 1]

    return {
        "spike_auc": spike_auc,
        "soma_explained_var": soma_var.item(),
        "dvt_correlation": dvt_corr.item(),
    }

def setup_datasets(config):
    files = glob(config.nmda_dataset.data_dir + "*_6_secDuration_*")
    valid_files = [random.choice(files)]
    train_files = [f for f in files if f not in valid_files]

    valid_files_per_epoch = max(1, int(config.training.validation_fraction *
                                     config.training.train_files_per_epoch))

    train_dataset = SimulationDataset(
        train_files, valid_files_per_epoch, config.training.window_size_ms,
        config.training.train_file_load, config.training.ignore_time_from_start,
        config.training.y_train_soma_bias, config.training.v_threshold,
        config.training.y_DTV_threshold, config.training.curr_file_index,
        is_shuffle=True
    )

    val_dataset = SimulationDataset(
        valid_files, len(valid_files), config.training.window_size_ms,
        config.training.train_file_load, config.training.ignore_time_from_start,
        config.training.y_train_soma_bias, config.training.v_threshold,
        config.training.y_DTV_threshold, config.training.curr_file_index,
        is_shuffle=False
    )

    return train_dataset, val_dataset

def create_data_loaders(train_dataset, val_dataset, config):
    return (
        DataLoader(train_dataset, batch_size=config.training.batch_size,
                  shuffle=False, num_workers=4, pin_memory=True),
        DataLoader(val_dataset, batch_size=config.training.batch_size,
                  shuffle=False, num_workers=4, pin_memory=True)
    )

def train_step(engine, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

    X_batch, (y_spike_batch, y_soma_batch, y_DVT_batch) = batch
    X_batch = X_batch.to(device, non_blocking=True)
    y_spike_batch = y_spike_batch.to(device, non_blocking=True)
    y_soma_batch = y_soma_batch.to(device, non_blocking=True)
    y_DVT_batch = y_DVT_batch.to(device, non_blocking=True)

    predictions = model(X_batch, dt=0.1)
    total_loss, loss_components = criterion(
        predictions,
        (y_spike_batch, y_soma_batch, y_DVT_batch),
        config.loss_weights_per_epoch[engine.state.epoch]
    )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    metrics = compute_metrics(predictions, (y_spike_batch, y_soma_batch, y_DVT_batch))

    # Clean up to free memory
    del X_batch, y_spike_batch, y_soma_batch, y_DVT_batch, predictions

    return {"loss": total_loss.item(), **loss_components, **metrics}

@torch.no_grad()
def validation_step(engine, batch):
    model.eval()
    X_batch, (y_spike_batch, y_soma_batch, y_DVT_batch) = batch
    X_batch = X_batch.to(device, non_blocking=True)
    y_spike_batch = y_spike_batch.to(device, non_blocking=True)
    y_soma_batch = y_soma_batch.to(device, non_blocking=True)
    y_DVT_batch = y_DVT_batch.to(device, non_blocking=True)

    predictions = model(X_batch, dt=0.1)
    total_loss, loss_components = criterion(
        predictions,
        (y_spike_batch, y_soma_batch, y_DVT_batch),
        config.loss_weights_per_epoch[engine.state.epoch]
    )

    metrics = compute_metrics(predictions, (y_spike_batch, y_soma_batch, y_DVT_batch))

    del X_batch, y_spike_batch, y_soma_batch, y_DVT_batch, predictions

    return {"loss": total_loss.item(), **loss_components, **metrics}

if __name__ == "__main__":
    train_dataset, val_dataset = setup_datasets(config)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)

    input_size = train_dataset[0][0].shape[-1]
    model = CSnT(
        input_size=input_size,
        hidden_size=config.model.hidden_size,
        output_sizes=config.model.output_sizes,
        num_layers=config.model.num_layers,
        nhead=config.model.nhead,
        dropout=config.model.dropout
    ).to(device)

    criterion = CustomLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate_per_epoch[0],
        weight_decay=config.weight_decay
    )

    trainer = Engine(train_step)
    evaluator = Engine(validation_step)

    # Attach metrics
    metrics = ["loss", "spike_loss", "soma_loss", "dvt_loss", "temporal_loss",
              "spike_auc", "soma_explained_var", "dvt_correlation"]

    for metric in metrics:
        RunningAverage(output_transform=lambda x, m=metric: x[m]).attach(trainer, metric)
        RunningAverage(output_transform=lambda x, m=metric: x[m]).attach(evaluator, metric)

    # Handlers
    ProgressBar().attach(trainer, ["loss", "spike_auc", "soma_explained_var"])
    ProgressBar().attach(evaluator, ["loss", "spike_auc", "soma_explained_var"])

    checkpoint_handler = ModelCheckpoint(
            get_experiment_dir(),
            "csnt_v2",
            n_saved=3,
            require_empty=False,
            score_function=lambda engine: -engine.state.metrics["loss"],
            score_name="loss",
            global_step_transform=lambda *_: trainer.state.iteration,
        )
    evaluator.add_event_handler(
        Events.COMPLETED, checkpoint_handler, {"model": model, "optimizer": optimizer}
    )

    evaluator.add_event_handler(
        Events.COMPLETED,
        EarlyStopping(
            patience=15,
            score_function=lambda engine: -engine.state.metrics["loss"],
            trainer=trainer
        )
    )

    @trainer.on(Events.EPOCH_STARTED)
    def reset_and_clean(engine):
        torch.cuda.empty_cache()
        gc.collect()
        model.reset_states()
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        for name, value in metrics.items():
            writer.add_scalar(f"validation/{name}", value, engine.state.epoch)
            print((f"validation/{name}", value, engine.state.epoch))
        torch.cuda.empty_cache()
        gc.collect()

    trainer.run(train_loader, max_epochs=config.num_epochs)
    writer.close()
