import math
import torch

import lightning as L
import torch.nn.functional as F


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


class SimpleLSTMLearning(L.LightningModule):
    '''LSTM Learning Model.'''

    def __init__(self, dict_encoder: Dict, latent_dim=32):
        super().__init__()
        self.dict_encoder = dict_encoder
        self.dict_encoder.freeze()
        self.time_encoder = SimpleLSTMAutoRegressiveModel(latent_dim=latent_dim)

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


class LSTMLearning(SimpleLSTMLearning):

    def __init__(self, dict_encoder: Dict, latent_dim=32):
        super().__init__(dict_encoder, latent_dim)
        self.time_encoder = LSTMAutoRegressiveModel(latent_dim=latent_dim)
