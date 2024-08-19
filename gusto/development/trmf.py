'''Temporal Regularized Matrix Factorization (TRMF) model for time series forecasting.'''

from pathlib import Path

import math
import torch

from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from gusto.lib.data import GustoFactorisedDataset
from gusto.lib.torch_lightning import LITModel


class PositionalEncoding(torch.nn.Module):
    '''PositionalEncoding for the AutoRegressiveModel.'''

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def _get_pe(self, pos: torch.Tensor) -> torch.Tensor:
        pe = torch.zeros(pos.size(0), pos.size(1), self.d_model).to(pos.device)
        pe[:, :, 0::2] = torch.sin(pos.unsqueeze(-1) * self.div_term)
        pe[:, :, 1::2] = torch.cos(pos.unsqueeze(-1) * self.div_term)
        return pe

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            time: Tensor, shape ``[batch_size, seq_len]
        '''
        pe = self._get_pe(pos)
        return x + pe

    def reverse(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        '''
        pe = self._get_pe(pos)
        return x - pe


class Encoder(torch.nn.Module):
    '''Encoder for the AutoRegressiveModel.'''

    def __init__(self, input_dim=383, latent_dim=64):
        super().__init__()
        self.forward_xfm = torch.nn.Linear(input_dim, latent_dim, bias=False)
        self.positional_encoding = PositionalEncoding(d_model=latent_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, input_dim]``
            pos: Tensor, shape ``[batch_size, seq_len]``
            mask: Tensor, shape ``[batch_size, seq_len, input_dim]``
        '''
        x = x * mask
        x = self.forward_xfm(x)
        x = self.positional_encoding(x, pos)
        return x


class Decoder(torch.nn.Module):
    '''Decoder for the AutoRegressiveModel.'''

    def __init__(self, input_dim=383, latent_dim=64):
        super().__init__()
        self.backward_xfm = torch.nn.Linear(latent_dim, input_dim, bias=False)
        self.positional_encoding = PositionalEncoding(d_model=latent_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, input_dim]``
        '''
        x = self.positional_encoding.reverse(x, pos)
        x = self.backward_xfm(x)
        return x * mask


class AutoRegressiveModel(torch.nn.Module):
    '''AutoRegressiveModel for time series forecasting.'''

    def __init__(self, input_dim=383, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.time_encoder = torch.nn.LSTM(latent_dim, latent_dim, batch_first=True, num_layers=1)
        self.decoder = Decoder(input_dim=input_dim, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, input_dim]``
            pos: Tensor, shape ``[batch_size, seq_len]``
            mask: Tensor, shape ``[batch_size, seq_len, input_dim]``
        '''
        x = self.encoder(x, pos, mask)
        _, (h, _) = self.time_encoder(x)
        return h.transpose(0, 1)


class LITAutoRegressiveModel(LITModel):
    '''Lightning module for the AutoRegressiveModel.'''

    def __init__(self, input_dim=383, latent_dim=64):
        super().__init__()
        self.val_metric2 = self.metric_func()
        self._lambda = 0
        self.model = AutoRegressiveModel(input_dim=input_dim, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return self.model(x, pos, mask)

    def training_step(self, train_batch, *args):
        inputs, outputs = train_batch
        out, pos, mask = outputs
        out_pred_enc = self(*inputs)
        out_gt_enc = self.model.encoder(*outputs)

        self.train_metric.update(out_pred_enc, out_gt_enc)
        # reg = self.model.encoder.forward_xfm.weight.norm(1) * self._lambda
        loss_enc = self.loss_func(out_pred_enc, out_gt_enc)
        loss_dec = self.loss_func(self.model.decoder(out_pred_enc, pos, mask), out)
        loss = loss_enc + loss_dec
        self.log('train_loss', loss)
        self.log('train_loss_enc', loss_enc)
        self.log('train_loss_dec', loss_dec)
        # self.log('train_reg', reg)
        # return loss_enc + loss_dec
        return loss_dec

    def validation_step(self, val_batch, *args):
        inputs, outputs = val_batch
        out, pos, mask = outputs
        out_pred_enc = self(*inputs)
        out_gt_enc = self.model.encoder(*outputs)
        out_pred = self.model.decoder(out_pred_enc, pos, mask)
        loss = self.loss_func(out_pred, out)

        if not self.trainer.sanity_checking:
            self.val_metric2.update(out_pred_enc, out_gt_enc)
            self.val_metric.update(out_pred, out)
        return loss

    def predict_step(self, batch):
        raise NotImplementedError

    def test_step(self, batch, *args):
        inputs, outputs = batch
        out, pos, mask = outputs
        out_pred_enc = self(*inputs)
        out_pred = self.model.decoder(out_pred_enc, pos, mask)

        self.test_metric.update(out_pred, out)
        return self.loss_func(out_pred, out)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.logger.experiment.add_scalars(
                'epoch_loss', {'validation': self.val_metric.compute()}, self.current_epoch
            )
            self.logger.experiment.add_scalars(
                'epoch_loss', {'validation_enc': self.val_metric2.compute()}, self.current_epoch
            )
            self.log('val_loss', self.val_metric.compute(), on_step=False)
            self.val_metric.reset()


def get_logger(test_name, latent_dim):
    '''Get the logger for the Lightning model'''
    logdir = Path.home().joinpath('Dev', 'git', 'gusto', 'logs', f'{test_name}_{latent_dim}')
    if not logdir.parent.is_dir():
        logdir.parent.mkdir(parents=True)
    return TensorBoardLogger(
        save_dir=str(logdir.parent.parent),
        name='logs',
        version=f'{test_name}_{latent_dim}',
    )


def get_model_checkpoint_callback(test_name, latent_dim):
    '''Get the model checkpoint callback for the Lightning model'''
    chkpoint_dir = Path.home().joinpath(
        'Dev', 'git', 'gusto', 'checkpoints', f'{test_name}_{latent_dim}'
    )
    if not chkpoint_dir.parent.is_dir():
        chkpoint_dir.parent.mkdir(parents=True)
    return ModelCheckpoint(
        dirpath=chkpoint_dir,
        filename='{epoch}-{val_loss:.5f}',
        save_top_k=3,
        monitor='val_loss',
    )


if __name__ == '__main__':
    tname = Path(__file__).stem
    latent_size = 64
    trainer = Trainer(
        max_epochs=300,
        logger=get_logger(tname, latent_size),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback(tname, latent_size)],
        log_every_n_steps=4,
    )
    train_dataset = GustoFactorisedDataset(
        Path.home().joinpath(
            'Dev', 'git', 'gusto', 'data', '3-9-48-72months_383CpGs_153indivs_train.pkl'
        )
    )
    val_dataset = GustoFactorisedDataset(
        Path.home().joinpath(
            'Dev', 'git', 'gusto', 'data', '3-9-48-72months_383CpGs_153indivs_val.pkl'
        )
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True
    )
    model = LITAutoRegressiveModel(latent_dim=latent_size)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
