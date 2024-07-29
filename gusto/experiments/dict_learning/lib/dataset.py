import pickle
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset


class DictionaryDataset(Dataset):
    '''Dataset for Dictionary Learning.'''

    def __init__(self, filepath, x_indices=None, y_indices=None):
        self._x_indices = x_indices if x_indices is not None else np.arange(0, 3)
        self._y_indices = y_indices
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        self.input_data = np.concatenate(
            [
                data_dict['data'][subj_id]['data'][:, self._x_indices].T
                for subj_id in data_dict['data']
            ],
            axis=0,
        )
        self.output_data = (
            np.concatenate(
                [
                    data_dict['data'][subj_id]['data'][:, self._y_indices].T
                    for subj_id in data_dict['data']
                ],
                axis=0,
            )
            if self._y_indices is not None
            else np.array([])
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.input_data, self.output_data


class DictionaryLSTMDataset(Dataset):
    '''Dataset where inputs are alpha coefficients and outputs are raw data.'''

    def __init__(
        self,
        num_subjects: int,
        alphas: torch.Tensor,
        filepath: Path,
        x_indices=None,
        y_indices=None,
    ):
        self._x_indices = x_indices if x_indices is not None else np.arange(0, 3)
        self._y_indices = y_indices if y_indices is not None else np.arange(3, 4)
        self.input_data = alphas.reshape(num_subjects, -1, alphas.shape[-1])[:, self._x_indices, :]
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        self.input_time = np.stack(
            [
                data_dict['data'][subj_id]['time'][self._x_indices, 0]
                for subj_id in data_dict['data']
            ],
            axis=0,
        )
        self.output_data = np.stack(
            [
                data_dict['data'][subj_id]['data'][:, self._y_indices].T
                for subj_id in data_dict['data']
            ],
            axis=0,
        )
        self.output_time = np.stack(
            [
                data_dict['data'][subj_id]['time'][self._y_indices, 0]
                for subj_id in data_dict['data']
            ],
            axis=0,
        )
        print('done')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return (self.input_data, self.input_time), (self.output_data, self.output_time)


class LSTMDataset(Dataset):
    '''Dataset for LSTM Learning.'''

    def __init__(self, alphas: torch.Tensor, x_indices=None, y_indices=None):
        self._x_indices = x_indices if x_indices is not None else np.arange(0, 3)
        self._y_indices = y_indices if y_indices is not None else np.arange(3, 4)
        alphas = alphas.reshape(-1, len(self._x_indices) + len(self._y_indices), alphas.shape[-1])
        self.input_data = alphas[:, self._x_indices, :]
        self.output_data = alphas[:, self._y_indices, :]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.input_data, self.output_data
