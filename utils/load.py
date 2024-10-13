import pickle
import time

import numpy as np


def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms):
    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype="bool")
    for row_ind, spike_times in row_inds_spike_times_map.items():
        bin_spikes_matrix[row_ind, spike_times] = 1.0
    return bin_spikes_matrix


def parse_sim_experiment_with_DVT(
    sim_experiment_file,
    y_train_soma_bias=-67.7,
    y_soma_threshold=-55.0,
    y_DTV_threshold=3.0,
    DVT_PCA_model=None,
    print_logs=False,
):

    with open(sim_experiment_file, "rb") as f:
        experiment_dict = pickle.load(f)

    num_simulations = len(experiment_dict["Results"]["listOfSingleSimulationDicts"])
    num_segments = len(experiment_dict["Params"]["allSegmentsType"])
    sim_duration_ms = int(experiment_dict["Params"]["totalSimDurationInSec"] * 1000)
    num_synapses = num_segments * 2  # num_ex_synapses + num_inh_synapses

    # Initialize
    X = np.zeros((num_synapses, sim_duration_ms, num_simulations), dtype="bool")
    y_spike = np.zeros((sim_duration_ms, num_simulations))
    y_soma = np.zeros((sim_duration_ms, num_simulations))

    if DVT_PCA_model is not None:
        y_DVTs = np.zeros(
            (DVT_PCA_model.n_components, sim_duration_ms, num_simulations),
            dtype=np.float32,
        )
    else:
        y_DVTs = np.zeros(
            (num_segments, sim_duration_ms, num_simulations), dtype=np.float16
        )


    for k, sim_dict in enumerate(
        experiment_dict["Results"]["listOfSingleSimulationDicts"]
    ):
        X[:, :, k] = np.vstack(
            (
                dict2bin(sim_dict["exInputSpikeTimes"], num_segments, sim_duration_ms),
                dict2bin(sim_dict["inhInputSpikeTimes"], num_segments, sim_duration_ms),
            )
        )

        spike_times = (sim_dict["outputSpikeTimes"].astype(float) - 0.5).astype(int)
        y_spike[spike_times, k] = 1.0
        y_soma[:, k] = sim_dict["somaVoltageLowRes"]

        curr_DVTs = np.clip(sim_dict["dendriticVoltagesLowRes"], 0, 2)
        if DVT_PCA_model is not None:
            y_DVTs[:, :, k] = DVT_PCA_model.transform(curr_DVTs.T).T
        else:
            y_DVTs[:, :, k] = curr_DVTs

    # Match the structure
    X = np.transpose(X, axes=[2, 0, 1])
    y_spike = y_spike.T[:, :, np.newaxis]
    y_soma = y_soma.T[:, :, np.newaxis]
    y_DVTs = np.transpose(y_DVTs, axes=[2, 0, 1])

    # threshold the signals
    y_soma[y_soma > y_soma_threshold] = y_soma_threshold
    y_DVTs[y_DVTs > y_DTV_threshold] = y_DTV_threshold
    y_DVTs[y_DVTs < -y_DTV_threshold] = -y_DTV_threshold

    y_soma = y_soma - y_train_soma_bias

    return X, y_spike, y_soma, y_DVTs
