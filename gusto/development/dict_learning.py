import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.utils.data import Dataset, DataLoader

from gusto.lib.data import TRAINING_DATA, VALIDATION_DATA
from gusto.development.trmf import Decoder, Encoder, get_logger


class Dict(L.LightningModule):
    '''Dictionary Learning Model.'''

    def __init__(self, latent_dim=32, input_dim=383):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.dict_encoder = torch.nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, input_dim]``
        '''
        return self.dict_encoder(x)


class Alpha(L.LightningModule):
    '''Alpha coefficients'''

    def __init__(self, input_num, latent_dim=32):
        super().__init__()
        self.alpha_params = torch.nn.Parameter(torch.randn(input_num, latent_dim))


class DictLearning(L.LightningModule):
    '''Dictionary Learning Model.'''

    def __init__(self, alpha, latent_dim=32, input_dim=383, lmbda=0.1, train_alpha_only=False):
        super().__init__()
        self.lmbda = lmbda
        self.input_dim = 383
        self.dict_encoder = Dict(latent_dim=latent_dim, input_dim=input_dim)
        self.alpha: Alpha = alpha
        self.automatic_optimization = False
        if train_alpha_only:
            self.dict_encoder.freeze()

    def configure_optimizers(self):
        dict_optimizer = torch.optim.Adam(self.dict_encoder.parameters())
        alpha_optimizer = torch.optim.Adam(self.alpha.parameters(), lr=1e-2)
        return dict_optimizer, alpha_optimizer

    def training_step(self, batch, *args):
        '''

        Args:
            batch: (inputs,): shape -> (batch, timestep, input_dim)

        '''
        # pylint: disable=arguments-differ
        input_data, _ = batch
        dict_opt: torch.optim.Adam
        alpha_opt: torch.optim.Adam
        dict_opt, alpha_opt = self.optimizers()  # type: ignore # pylint: disable=unpacking-non-sequence
        inp = input_data.view(-1, self.input_dim)

        # Optimise alpha
        alpha_pred = self.dict_encoder(self.alpha.alpha_params)
        reg = self.lmbda * torch.norm(self.alpha.alpha_params, p=1)
        alpha_loss = F.mse_loss(alpha_pred, inp) + reg

        alpha_opt.zero_grad()
        self.manual_backward(alpha_loss)
        alpha_opt.step()

        # Optimise dict
        dict_pred = self.dict_encoder(self.alpha.alpha_params)
        # TODO: maybe add l2 regularisation for dict to prevent D 2-norm from greater than 1
        dict_loss = F.mse_loss(dict_pred, inp)

        dict_opt.zero_grad()
        self.manual_backward(dict_loss)
        dict_opt.step()

        self.log_dict({'train_loss_dict': dict_loss, 'train_loss_alpha': alpha_loss}, prog_bar=True)


class SimpleLSTMAutoRegressiveModel(L.LightningModule):
    '''Simple LSTM AutoRegressive Model.'''

    def __init__(self, latent_dim=64):
        super().__init__()
        self.time_encoder = torch.nn.LSTM(latent_dim, latent_dim, batch_first=True, num_layers=1)

    def forward(self, x: torch.Tensor, pos_in: torch.Tensor, pos_out: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len_in, input_dim]``
            pos_in: Tensor, shape ``[batch_size, seq_len_in]``
            pos_out: Tensor, shape ``[batch_size, seq_len_out]``
        '''
        _, (h, _) = self.time_encoder(x)
        return h.transpose(0, 1)


class LSTMAutoRegressiveModel(L.LightningModule):
    '''LSTM AutoRegressive Model.'''

    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(input_dim=latent_dim, latent_dim=latent_dim)
        self.time_encoder = torch.nn.LSTM(latent_dim, latent_dim, batch_first=True, num_layers=1)
        self.decoder = Decoder(input_dim=latent_dim, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor, pos_in: torch.Tensor, pos_out: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len_in, input_dim]``
            pos_in: Tensor, shape ``[batch_size, seq_len_in]``
            pos_out: Tensor, shape ``[batch_size, seq_len_out]``
        '''
        x = self.encoder(x, pos_in, 1)
        _, (h, _) = self.time_encoder(x)
        x = self.decoder(h.transpose(0, 1), pos_out, 1)
        return x


class LSTMLearning(L.LightningModule):
    '''LSTM Learning Model.'''

    def __init__(self, dict_encoder: Dict, latent_dim=32):
        super().__init__()
        self.dict_encoder = dict_encoder
        self.dict_encoder.freeze()
        # self.time_encoder = SimpleLSTMAutoRegressiveModel(latent_dim=latent_dim)
        self.time_encoder = LSTMAutoRegressiveModel(latent_dim=latent_dim)

    def configure_optimizers(self):
        lstm_optimizer = torch.optim.Adam(self.time_encoder.parameters())
        return lstm_optimizer

    def training_step(self, batch, *args):
        '''

        Args:
            batch: (inputs,): shape -> (batch, timestep, input_dim)

        '''
        # pylint: disable=arguments-differ
        (input_data, input_time), (output_data, output_time) = batch
        b, s, t_in, l_in = input_data.shape
        _, _, t_out, l_out = output_data.shape
        input_data = input_data.view(b * s, t_in, l_in)
        input_time = input_time.view(b * s, t_in)
        output_data = output_data.view(b * s, t_out, l_out)
        output_time = output_time.view(b * s, t_out)
        pred_alphas = self.time_encoder(input_data, input_time, output_time)
        pred_out_data = self.dict_encoder(pred_alphas)
        training_loss = torch.sqrt(F.mse_loss(pred_out_data, output_data))
        self.log('train_loss_lstm', training_loss, prog_bar=True)
        return training_loss

    def validation_step(self, batch, *args):
        # pylint: disable=arguments-differ
        (input_data, input_time), (output_data, output_time) = batch
        b, s, t, l_in = input_data.shape
        _, _, t_out, l_out = output_data.shape
        input_data = input_data.view(b * s, t, l_in)
        input_time = input_time.view(b * s, t)
        output_data = output_data.view(b * s, t_out, l_out)
        output_time = output_time.view(b * s, t_out)
        pred_alphas = self.time_encoder(input_data, input_time, output_time)
        pred_out_data = self.dict_encoder(pred_alphas)
        val_loss = torch.sqrt(F.mse_loss(pred_out_data, output_data))
        self.log('val_loss_lstm', val_loss)
        return val_loss


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


def get_model_checkpoint_callback(name, latent_dim, filename, monitor='train_loss_alpha'):
    '''Get the model checkpoint callback for the Lightning model'''
    chkpoint_dir = Path.home().joinpath(
        'Dev', 'git', 'gusto', 'checkpoints', f'{name}_{latent_dim}'
    )
    if not chkpoint_dir.parent.is_dir():
        chkpoint_dir.parent.mkdir(parents=True)
    return ModelCheckpoint(dirpath=chkpoint_dir, filename=filename, save_top_k=1, monitor=monitor)


def run_step_one(latent_dim, test_name):
    '''Step one learns the dictionary D and training input & output alphas'''
    dataset = DictionaryDataset(TRAINING_DATA, x_indices=[0, 1, 2, 3])
    train_alpha = Alpha(input_num=dataset.input_data.shape[0], latent_dim=latent_dim)
    model = DictLearning(train_alpha, latent_dim, train_alpha_only=False)
    trainer = L.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=1,
        logger=get_logger(f'{test_name}_step1', latent_dim),
        callbacks=[get_model_checkpoint_callback(f'{test_name}', latent_dim, 'step_one')],
    )
    trainer.fit(model, DataLoader(dataset))
    print('Finished training')


def run_step_two(latent_dim, test_name):
    '''Step two freezes D and learns learns the validation input alpha'''
    dataset = DictionaryDataset(VALIDATION_DATA, x_indices=[0, 1, 2])
    train_alpha = Alpha(input_num=488, latent_dim=latent_dim)
    val_alpha = Alpha(input_num=dataset.input_data.shape[0], latent_dim=latent_dim)
    model = DictLearning.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=f'checkpoints/{test_name}_{latent_dim}/step_one.ckpt',
        alpha=train_alpha,
        latent_dim=latent_dim,
        train_alpha_only=True,
    )
    model.alpha = val_alpha
    trainer = L.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=1,
        logger=get_logger(f'{test_name}_step2', latent_dim),
        callbacks=[get_model_checkpoint_callback(test_name, latent_dim, 'step_two')],
    )
    trainer.fit(model, DataLoader(dataset))


def run_step_three(latent_dim, test_name):
    '''Step three trains an LSTM model with the frozen train dictionary D and alpha coefficients,
    and validates against frozen input validation alpha coefficients
    '''
    train_dict = DictLearning.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=f'checkpoints/{test_name}_{latent_dim}/step_one.ckpt',
        alpha=Alpha(input_num=488, latent_dim=latent_dim),
        latent_dim=latent_dim,
        train_alpha_only=False,
    )
    val_dict = DictLearning.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=f'checkpoints/{test_name}_{latent_dim}/step_two.ckpt',
        alpha=Alpha(input_num=45, latent_dim=latent_dim),
        latent_dim=latent_dim,
        train_alpha_only=True,
    )
    train_dataset = DictionaryLSTMDataset(
        122, train_dict.alpha.alpha_params.detach(), TRAINING_DATA
    )
    val_dataset = DictionaryLSTMDataset(15, val_dict.alpha.alpha_params.detach(), VALIDATION_DATA)
    model = LSTMLearning(train_dict.dict_encoder, latent_dim=latent_dim)
    trainer = L.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=1,
        logger=get_logger(f'{test_name}_step3_ae', latent_dim),
        callbacks=[
            get_model_checkpoint_callback(
                test_name, latent_dim, 'step_three_ae', monitor='val_loss_lstm'
            )
        ],
    )
    trainer.fit(model, DataLoader(train_dataset), DataLoader(val_dataset))


if __name__ == '__main__':
    latent_size = 256
    tname = f'{Path(__file__).stem}'
    run_step_one(latent_size, tname)
    run_step_two(latent_size, tname)
    run_step_three(latent_size, tname)
