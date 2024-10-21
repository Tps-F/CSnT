from pathlib import Path

import numpy as np
import torch
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader, Dataset

from utils.load import parse_sim_experiment_with_DVT


class SimulationDataset(Dataset):
    def __init__(
        self,
        sim_experiment_files,
        num_files_per_epoch=6,
        window_size_ms=300,
        file_load=0.3,
        ignore_time_from_start=500,
        y_train_soma_bias=-67.7,
        y_soma_threshold=-55.0,
        y_DTV_threshold=3.0,
        curr_file_index=-1,
        is_shuffle=True,
    ):
        self.sim_experiment_files = sim_experiment_files
        self.num_files_per_epoch = num_files_per_epoch
        self.window_size_ms = window_size_ms
        self.file_load = file_load
        self.ignore_time_from_start = ignore_time_from_start
        self.y_train_soma_bias = y_train_soma_bias
        self.y_soma_threshold = y_soma_threshold
        self.y_DTV_threshold = y_DTV_threshold
        self.curr_file_index = curr_file_index
        self.is_shuffle = is_shuffle

        self.shuffle()
        self.load_file()

        self.num_simulations_per_file, self.sim_duration_ms, self.num_segments = (
            self.X.shape
        )
        self.num_output_channels_y1 = self.y_spike.shape[2]
        self.num_output_channels_y2 = self.y_soma.shape[2]
        self.num_output_channels_y3 = self.y_DVT.shape[2]

        self.max_batches_per_file = (
            self.num_simulations_per_file * self.sim_duration_ms
        ) / self.window_size_ms
        self.batches_per_file = int(self.file_load * self.max_batches_per_file)
        self.batches_per_epoch = self.batches_per_file * self.num_files_per_epoch

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, batch_idx):
        if (batch_idx + 1) % self.batches_per_file == 0:
            self.load_file()

        sim_ind = np.random.choice(self.num_simulations_per_file, replace=True)
        sampling_start_time = max(self.ignore_time_from_start, self.window_size_ms)
        win_time = np.random.choice(
            range(sampling_start_time, self.sim_duration_ms),
            replace=False,
        )

        X = np.zeros((self.window_size_ms, self.num_segments))
        y_spike = np.zeros((self.window_size_ms, self.num_output_channels_y1))
        y_soma = np.zeros((self.window_size_ms, self.num_output_channels_y2))
        y_DVT = np.zeros((self.window_size_ms, self.num_output_channels_y3))

        X = self.X[sim_ind, win_time - self.window_size_ms : win_time]
        y_spike = self.y_spike[sim_ind, win_time - self.window_size_ms : win_time]
        y_soma = self.y_soma[sim_ind, win_time - self.window_size_ms : win_time]
        y_DVT = self.y_DVT[sim_ind, win_time - self.window_size_ms : win_time]

        return torch.tensor(X).float(), (
            torch.tensor(y_spike).float(),
            torch.tensor(y_soma).float(),
            torch.tensor(y_DVT).float(),
        )

    def shuffle(self):
        self.curr_epoch_files_to_use = (
            np.random.choice(
                self.sim_experiment_files, self.num_files_per_epoch, replace=False
            )
            if self.is_shuffle
            else self.sim_experiment_files[: self.num_files_per_epoch]
        )

    def load_file(self):
        self.curr_file_index = (self.curr_file_index + 1) % self.num_files_per_epoch
        self.curr_file_in_use = self.curr_epoch_files_to_use[self.curr_file_index]
        self.X, self.y_spike, self.y_soma, self.y_DVT = parse_sim_experiment_with_DVT(
            self.curr_file_in_use
        )
