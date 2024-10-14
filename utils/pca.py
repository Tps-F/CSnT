import numpy
import torch


def pca_torch(data, num_components):

    data_mean = torch.mean(data, dim=0)
    centered_data = data - data_mean

    cov_matrix = torch.mm(centered_data.t(), centered_data) / (
        centered_data.shape[0] - 1
    )

    eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix, eigenvectors=True)
    eigenvalues = eigenvalues[:, 0]
    sorted_indices = torch.argsort(eigenvalues, descending=True)

    selected_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]

    total_variance = torch.sum(eigenvalues[sorted_indices])
    explained_variance_ratio = (
        eigenvalues[sorted_indices[:num_components]] / total_variance
    )

    return selected_eigenvectors, explained_variance_ratio
