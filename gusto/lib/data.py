'''Dataset classes for GUSTO data'''

import math
from pathlib import Path
import pickle

from typing import Dict

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from torch.utils.data import Dataset, DataLoader

import lightning as L

DATA_DIR = Path(__file__).parent.parent.parent.joinpath('data')

TRAINING_DATA = DATA_DIR.joinpath('3-9-48-72months_383CpGs_153indivs_train.pkl')
TRAINING_DATA_NOMISSING = DATA_DIR.joinpath('3-9-48-72months_383CpGs_153indivs_train_nomissing.pkl')
VALIDATION_DATA = DATA_DIR.joinpath('3-9-48-72months_383CpGs_153indivs_val.pkl')
VALIDATION_DATA_NOMISSING = DATA_DIR.joinpath('3-9-48-72months_383CpGs_153indivs_val_nomissing.pkl')
TEST_DATA = DATA_DIR.joinpath('3-9-48-72months_383CpGs_153indivs_test.pkl')

FULL_3M_DATA = DATA_DIR.joinpath(
    'EPIC850k-GUSTO-Buccal-3m-760446probes-202samples-Noob-BMIQ-beta.txt'
)
FULL_9M_DATA = DATA_DIR.joinpath(
    'EPIC850k-GUSTO-Buccal-9m-760446probes-314samples-Noob-BMIQ-beta.txt'
)
FULL_48M_DATA = DATA_DIR.joinpath(
    'EPIC850k-GUSTO-Buccal-48m-760446probes-336samples-Noob-BMIQ-beta.txt'
)


class GustoDataset(Dataset):
    '''Dataset class for GUSTO data'''

    def __init__(self, filepath, cpg_samples=100, x_indices=None, y_indices=None):
        self.cpg_samples = cpg_samples
        self.rng_state = np.random.default_rng(42)
        self._x_indices = x_indices if x_indices is not None else np.arange(0, 3)
        self._y_indices = y_indices if y_indices is not None else np.arange(3, 4)
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        self.data = data_dict['data']
        self.subject_ids = list(data_dict['data'].keys())
        self.probe_ids = data_dict['probeids']
        self._timepoints = data_dict['months'].astype(np.float32)
        self._normalise_timepoints()
        self.site_indices = self._get_probe_array()

    def _get_probe_array(self):
        indices = np.arange(len(self.probe_ids))
        shuffled = self.rng_state.choice(indices, size=len(self.probe_ids), replace=False)
        # Split the shuffled indices into cpg_samples-sized chunks
        out = [
            shuffled[i : i + self.cpg_samples] for i in range(0, len(shuffled), self.cpg_samples)
        ]

        # If the last site has fewer than cpg_samples, shuffle again and add to the last site
        if len(out[-1]) < self.cpg_samples:
            last = out.pop()
            indices = np.setdiff1d(indices, last)
            shuffled = self.rng_state.choice(
                indices, size=self.cpg_samples - len(last), replace=False
            )
            out.append(np.concatenate((last, shuffled)))
        return out

    def _normalise_timepoints(self):
        min_time = 0  # Set 0 as the starting point
        max_time = np.max(self._timepoints)
        self._timepoints = ((self._timepoints - min_time) / (max_time - min_time)).astype(
            np.float32
        )

    def __len__(self):
        return self.num_subjects * self.num_site_samples

    def _get_subject_data(self, idx):
        subject_key = self.subject_ids[idx // self.num_site_samples]

        data_dict = self.data[subject_key]
        data: np.ndarray = data_dict['data'].astype(np.float32)
        mask: np.ndarray = data_dict['mask'].astype(np.float32)
        return data, mask

    def _get_sites(self, idx):
        site_idx = idx % self.num_site_samples
        return self.site_indices[site_idx]

    def __getitem__(self, idx):
        data, _ = self._get_subject_data(idx)
        sites = self._get_sites(idx)
        x_indices = self.x_indices
        y_indices = self.y_indices
        input_idx_array = np.ix_(sites, x_indices)
        target_idx_array = np.ix_(sites, y_indices)
        time_in = self.timepoints[x_indices]
        time_out = self.timepoints[y_indices]
        input_data = data[input_idx_array].T[:, np.newaxis, :]
        target_data = data[target_idx_array].T[:, np.newaxis, :]

        return (input_data, time_in, time_out), target_data

    def shuffle_cpg_sites(self):
        '''Shuffle the CPG sites in the dataset'''
        self.site_indices = self._get_probe_array()

    @property
    def x_indices(self):
        '''Indices of the input timepoints in the dataset'''
        return self._x_indices

    @property
    def y_indices(self):
        '''Indices of the target timepoints in the dataset'''
        return self._y_indices

    @property
    def timepoints(self):
        '''Normalised timepoints in the dataset'''
        return self._timepoints

    @property
    def num_subjects(self):
        '''Number of subjects in the dataset'''
        return len(self.data)

    @property
    def num_sites(self):
        '''Number of CPG sites in the dataset'''
        return len(self.probe_ids)

    @property
    def num_site_samples(self):
        '''Number of `self.cpg_samples`-sized samples in the dataset'''
        return math.ceil(self.num_sites / self.cpg_samples)

    @property
    def num_timepoints(self):
        '''Number of timepoints in the dataset'''
        return len(self._timepoints)


class GustoFactorisedDataset(GustoDataset):
    '''Factorised dataset class for GUSTO data.'''

    def __init__(self, filepath, x_indices=None, y_indices=None):
        super().__init__(filepath, 383, x_indices, y_indices)
        print(f'Probe IDs: {self.probe_ids[16:26]}')

    def __getitem__(self, idx):
        data, mask = self._get_subject_data(idx)
        time_in = self.timepoints[self.x_indices]
        time_out = self.timepoints[self.y_indices]
        input_data = data[:, self.x_indices].T - 0.5
        input_mask = mask[:, self.x_indices].T
        target_data = data[:, self.y_indices].T - 0.5
        target_mask = mask[:, self.y_indices].T

        return (input_data, time_in, input_mask), (target_data, time_out, target_mask)

    def _normalise_timepoints(self):
        pass  # Do not need this method for the factorised dataset

    def _get_probe_array(self):
        pass  # Do not need this method for the factorised dataset

    @property
    def num_site_samples(self):
        '''Number of `self.cpg_samples`-sized samples in the dataset'''
        return 1


class GustoInterpDataset(GustoDataset):
    '''Interpolated dataset class for GUSTO data.

    Interpolates data across time and samples noisily (across time) the timepoints.
    '''

    def __init__(self, filepath, cpg_samples=100, interp_size=300, x_indices=None, y_indices=None):
        super().__init__(filepath, cpg_samples, x_indices, y_indices)
        self.rng_state = np.random.default_rng(42)
        self.interp_size = interp_size
        self.data = self._interpolate_data(self.data)

    def _interpolate_data(self, data: Dict[str, Dict[str, np.ndarray]]):
        timepoints = np.linspace(
            0, np.max(self._timepoints) * 1.1, self.interp_size, dtype=np.float32
        )
        for subject in data:
            data[subject]['data'] = self._interpolate_subject(data[subject]['data'], timepoints)
        self._interp_timepoints = timepoints
        self._orig_timepoint_idx = np.searchsorted(self._interp_timepoints, self._timepoints)
        self._x_interp_idx = self._orig_timepoint_idx[self._x_indices]
        self._y_interp_idx = self._orig_timepoint_idx[self._y_indices]
        return data

    def _interpolate_subject(self, data: np.ndarray, timepoints: np.ndarray) -> np.ndarray:
        out = np.zeros((data.shape[0], self.interp_size), dtype=np.float32)
        for i, site in enumerate(data):
            interp = Akima1DInterpolator(self._timepoints, site)
            out[i] = interp(timepoints, extrapolate=True)
        return out

    @property
    def x_indices(self):
        '''Indices of the input timepoints in the dataset'''
        idx_noise = np.round(
            self.rng_state.normal(loc=0, scale=3, size=len(self._x_indices))
        ).astype(int)

        idx_noise = np.clip(
            idx_noise,
            -self._x_interp_idx[0],
            len(self._interp_timepoints) - self._x_interp_idx[-1] - 1,
        )
        return self._x_interp_idx + idx_noise

    @property
    def y_indices(self):
        '''Indices of the target timepoints in the dataset'''
        idx_noise = np.round(
            self.rng_state.normal(loc=0, scale=3, size=len(self._y_indices))
        ).astype(int)
        idx_noise = np.clip(
            idx_noise,
            -self._y_interp_idx[0],
            len(self._interp_timepoints) - self._y_interp_idx[-1] - 1,
        )
        return self._y_interp_idx + idx_noise

    @property
    def timepoints(self):
        '''Normalised timepoints in the dataset'''
        return self._interp_timepoints


class GustoInterpDataModule(L.LightningDataModule):
    '''Lightning DataModule for the interpolated GUSTO dataset'''

    def __init__(
        self,
        train_filepath,
        val_filepath,
        cpg_samples=100,
        interp_size=300,
        x_indices=None,
        y_indices=None,
    ) -> None:
        super().__init__()
        self.train_filepath = train_filepath
        self.val_filepath = val_filepath
        self.cpg_samples = cpg_samples
        self.interp_size = interp_size
        self.x_indices = x_indices
        self.y_indices = y_indices
        self.train_dataset: GustoInterpDataset
        self.val_dataset: GustoDataset

    def setup(self, stage):
        self.train_dataset = GustoInterpDataset(
            self.train_filepath,
            cpg_samples=self.cpg_samples,
            interp_size=self.interp_size,
            x_indices=self.x_indices,
            y_indices=self.y_indices,
        )
        self.val_dataset = GustoDataset(
            self.val_filepath,
            cpg_samples=self.cpg_samples,
            x_indices=self.x_indices,
            y_indices=self.y_indices,
        )

    def train_dataloader(self):
        self.train_dataset.shuffle_cpg_sites()
        return DataLoader(
            self.train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True
        )


class GustoPreciseTimeDataset(GustoDataset):

    def __init__(self, filepath, cpg_samples=100, x_indices=None, y_indices=None):
        # pylint: disable=super-init-not-called
        super().__init__(filepath, cpg_samples, x_indices, y_indices)
        self._timepoints = None

    def _normalise_timepoints(self):
        pass  # Do not need this method for the precise time dataset

    def _get_subject_data(self, idx):
        subject_key = self.subject_ids[idx // self.num_site_samples]

        data_dict = self.data[subject_key]
        data: np.ndarray = data_dict['data'].astype(np.float32)
        mask: np.ndarray = data_dict['mask'].astype(np.float32)
        timepoints: np.ndarray = data_dict['time'][:, 0].astype(np.float32)
        return data, mask, timepoints

    def __getitem__(self, idx):
        data, _, timepoints = self._get_subject_data(idx)
        sites = self._get_sites(idx)
        x_indices = self.x_indices
        y_indices = self.y_indices
        input_idx_array = np.ix_(sites, x_indices)
        target_idx_array = np.ix_(sites, y_indices)
        time_in = timepoints[x_indices]
        time_out = timepoints[y_indices]
        input_data = data[input_idx_array].T[:, np.newaxis, :]
        target_data = data[target_idx_array].T[:, np.newaxis, :]

        return (input_data, time_in, time_out), target_data


class GustoSingleCpGDataset(GustoDataset):
    '''Dataset class for GUSTO data where each example is a single GpG site'''

    def __init__(self, filepath, x_indices=None, y_indices=None):
        super().__init__(filepath, cpg_samples=1, x_indices=x_indices, y_indices=y_indices)

    def _get_probe_array(self):
        indices = np.arange(len(self.probe_ids))
        shuffled = self.rng_state.choice(indices, size=len(self.probe_ids), replace=False)
        return shuffled

    def _get_sites(self, idx):
        site_idx = idx % self.num_site_samples
        return self.site_indices[site_idx : site_idx + 1]

    def __getitem__(self, idx):
        data, _ = self._get_subject_data(idx)
        sites = self._get_sites(idx)
        x_indices = self.x_indices
        y_indices = self.y_indices
        input_idx_array = np.ix_(sites, x_indices)
        target_idx_array = np.ix_(sites, y_indices)
        time_in = self.timepoints[x_indices]
        time_out = self.timepoints[y_indices]
        input_data = data[input_idx_array].squeeze(0) - 0.5
        target_data = data[target_idx_array].squeeze(0) - 0.5

        return (input_data, time_in, time_out), target_data

    @property
    def num_site_samples(self):
        '''Number of `self.cpg_samples`-sized samples in the dataset'''
        return self.num_sites


class GustoSingleCpGDatasetZeroMissing(GustoSingleCpGDataset):
    '''GUSTO dataset with missing values removed'''

    def __init__(self, filepath, x_indices=None, y_indices=None):
        super().__init__(filepath, x_indices=x_indices, y_indices=y_indices)
        self.rng_state = np.random.default_rng(42)
        self._x_indices = x_indices if x_indices is not None else np.arange(0, 3)
        self._y_indices = y_indices if y_indices is not None else np.arange(3, 4)
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        self.data = data_dict['data']
        self.subject_ids = list(data_dict['data'].keys())
        self.probe_ids = data_dict['probeids']
        self._timepoints = data_dict['months'].astype(np.float32)
        self._normalise_timepoints()
        self._remove_missing_data()

    def __len__(self):
        return self._len

    def _remove_missing_data(self):
        self._len = 0
        self._data_list = []
        for subject in self.data:
            data: np.ndarray = self.data[subject]['data']
            mask: np.ndarray = self.data[subject]['mask']
            valid_probe_ids = mask.sum(axis=-1) == 4
            self._len += valid_probe_ids.sum()
            self._data_list.append(data[valid_probe_ids])
        self._data = np.concatenate(self._data_list, axis=0) - 0.5

    def __getitem__(self, idx):
        time_in = self.timepoints[self.x_indices]
        time_out = self.timepoints[self.y_indices]
        input_data = self._data[idx, self.x_indices]
        target_data = self._data[idx, self.y_indices]

        return (input_data, time_in, time_out), target_data


if __name__ == '__main__':

    dataset = GustoPreciseTimeDataset(TRAINING_DATA)
    print(f'Number of subjects: {dataset.num_subjects}')
    print(f'First example: {dataset[0]}')
