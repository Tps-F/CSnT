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

files = glob(config.nmda_dataset.data_dir + "*_6_secDuration_*")[:1]

valid_files = random.choice(files)

train_files = [f for f in files if f != valid_files]

data_dict = {"train_files": train_files, "valid_files": valid_files}

_, _, _, y_DVTs = parse_sim_experiment_with_DVT(train_files[0])
X_pca_DVT = torch.tensor(np.reshape(y_DVTs, [y_DVTs.shape[0], -1]), dtype=torch.float32)

num_DVT_components = config.nmda_dataset.num_DVT_components

eigenvectors, explained_variance_ratio = pca_torch(
    X_pca_DVT, config.nmda_dataset.num_DVT_components
)
total_explained_variance = 100 * explained_variance_ratio.sum().item()
print(f"Total Explained Variance: {total_explained_variance:.2f}%")

X_train, y_spike_train, y_soma_train, y_DVT_train = (
    parse_multiple_sim_experiment_with_DVT(
        train_files,
        DVT_PCA_model=config.nmda_dataset.num_DVT_components,
        fit_structure=False,
    )
)

y_DVT_train[y_DVT_train > config.nmda_dataset.num_DVT_components] = (
    config.nmda_dataset.num_DVT_components
)
y_DVT_train[y_DVT_train < -config.nmda_dataset.num_DVT_components] = (
    -config.nmda_dataset.num_DVT_components
)

y_soma_train[y_soma_train > config.training.v_threshold] = config.training.v_threshold

sim_duration_ms = y_soma_train.shape[0]
sim_duration_sec = float(sim_duration_ms) / 1000

num_simulations_train = X_train.shape[-1]
