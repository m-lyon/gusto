'''UNet dataset'''

from math import ceil, prod

from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from gusto.lib.data import FULL_3M_DATA, FULL_9M_DATA, FULL_48M_DATA


class UNetDataset(Dataset):
    '''Dataset for UNet training.'''

    def __init__(
        self,
        start: float = 0.0,
        end: float = 0.9,
        dims=(256, 256),
        strategy='valid',
        x_indices=None,
        y_indices=None,
    ):
        '''

        Args:
            start: start of the dataset as a percentage
            end: end of the dataset as a percentage
            dims: dimension to reshape the CpG sites into
            strategy: strategy for reshaping data into images, valid options are:
                `valid`, `single`, and `wrap`. `valid` will extract `N` images from the
                dataset where `N` is the number of times the image dimension fits into the
                dataset with no wraparound. `single` will use the first image of dimension
                `dims` and `wrap` will use all CpG sites, and pad with other CpG sites if
                necessary.

        '''
        super().__init__()
        if strategy not in ['valid', 'single', 'wrap']:
            raise ValueError('Invalid strategy.')
        self.rng = np.random.default_rng(42)
        self._x_indices = x_indices if x_indices is not None else np.arange(0, 2)
        self._y_indices = y_indices if y_indices is not None else np.arange(2, 3)
        self.data, self.mask, self.months = self._load_data(start, end, dims, strategy)

    def _load_data(
        self, start: float, end: float, dims: Tuple[int, int], strategy: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        print('Loading data...')
        dfs = {
            '3': pd.read_csv(FULL_3M_DATA, sep='\t', index_col=0),
            '9': pd.read_csv(FULL_9M_DATA, sep='\t', index_col=0),
            '48': pd.read_csv(FULL_48M_DATA, sep='\t', index_col=0),
        }
        subj_ids = self._get_subject_ids(start, end, *dfs.values())
        print(f'Found {len(subj_ids)} subjects.')
        probe_ids = self._get_probe_ids(*dfs.values())
        print(f'Found {len(probe_ids)} CpG sites.')
        data, months = self._concat_data(subj_ids, probe_ids, dfs)
        print('Reshaping data...')
        data = self._reshape_data(data, dims, strategy)
        data, mask = self._get_mask(data)
        print('Done.')
        return data, mask, months

    def _get_subject_ids(self, start: float, end: float, *dfs: pd.DataFrame):
        subj_sets = []
        for df in dfs:
            # use regex to remove the prefix for each header, which begins with 'M3-', 'M9-', etc.
            columns = df.columns.str.replace(r'^M[0-9]+-', '', regex=True)
            subj_sets.append(set(columns))
        subj_ids = list(set.intersection(*tuple(subj_sets)))
        self.rng.shuffle(subj_ids)
        subj_ids = subj_ids[int(len(subj_ids) * start) : int(len(subj_ids) * end)]
        return subj_ids

    def _get_probe_ids(self, *dfs: pd.DataFrame):
        probe_id_sets = []
        for df in dfs:
            probe_id_sets.append(set(df.index))
        probe_ids = list(set.intersection(*tuple(probe_id_sets)))
        self.rng.shuffle(probe_ids)
        return probe_ids

    def _concat_data(
        self, subj_ids: List[str], probe_ids: List[str], dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, List[str]]:
        arrays = []
        for month, df in dfs.items():
            arr = df.loc[probe_ids, [f'M{month}-{subj_id}' for subj_id in subj_ids]].to_numpy()
            arrays.append(arr)
        data = np.stack(arrays, axis=-1)
        return data, list(dfs.keys())

    def _reshape_data(self, data: np.ndarray, dims: Tuple[int, int], strategy: str) -> np.ndarray:
        image_size = prod(dims)
        if strategy == 'valid':
            num = data.shape[0] // image_size
            data = data[: num * image_size, ...]

        elif strategy == 'single':
            num = 1
            data = data[:image_size, ...]
            data = data.reshape(1, *dims, -1, data.shape[-1])
        elif strategy == 'wrap':
            num = ceil(data.shape[0] / image_size)
            if num > (data.shape[0] // image_size):
                wrapped_data = data[:image_size, ...]
                data = np.concatenate([data, wrapped_data], axis=0)
            data = data[: num * image_size, ...]

        data = data.reshape(num, *dims, -1, data.shape[-1])
        data = data.transpose(0, 3, 4, 1, 2)
        data = data.reshape(-1, data.shape[2], *dims)
        return data.astype(np.float32)

    def _get_mask(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = (np.isnan(data) | (data == 0)).astype(np.float32)
        data = np.nan_to_num(data)
        return data, mask

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        '''Return the input and output data for the dataset.

        Args:
            idx: index of the dataset

        Returns:
            inp: input data, shape ``(inp_time, 1, h, w)``
            out: output data, shape ``(out_time, 1, h, w)``
        '''
        input_data = self.data[idx, self._x_indices, None, ...]
        input_mask = self.mask[idx, self._x_indices, None, ...]
        output_data = self.data[idx, self._y_indices, None, ...]
        # output_mask = self.mask[idx, self._y_indices, None, ...]
        return (input_data, input_mask), output_data


if __name__ == '__main__':
    dataset = UNetDataset(start=0.0, end=1.0)
    print(len(dataset))
    print(dataset[0][0][0].shape, dataset[0][1].shape)
