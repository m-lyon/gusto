'''Third UNet experiment with 2D data.

This experiment trains a mean and variance model.
'''

from pathlib import Path
from typing import Tuple

import torch

import lightning as L

from gusto.lib.torch_lightning import LITModel
from gusto.lib.utils import get_logger_reruns, get_model_checkpoint_callback

from gusto.experiments.unet.lib.utils import LOGDIR, CHECKPOINT_DIR
from gusto.experiments.unet.experiment_1 import UNetEncoder, UNetDecoder, get_dataset


torch.set_float32_matmul_precision('high')

WARMUP_EPOCHS = 25


class AleotoricUNet(LITModel):
    '''Aleotoric UNet model.'''

    def __init__(self):
        super().__init__()
        self.mean_decoder = UNetDecoder()
        self.variance_decoder = UNetDecoder()
        self.mean_encoder = UNetEncoder()
        self.variance_encoder = UNetEncoder()
        # Freeze variance weights
        for param in self.variance_encoder.parameters():
            param.requires_grad = False
        for param in self.variance_decoder.parameters():
            param.requires_grad = False

    @property
    def loss_func(self):
        return torch.nn.GaussianNLLLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-3)
        return optimizer

    def training_step(self, train_batch, *args):
        inputs, target = train_batch
        pred_mean, pred_var = self(*inputs)

        self.train_metric.update(pred_mean, target)
        loss = self.loss_func(pred_mean, target, pred_var)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, *args):
        inputs, target = val_batch
        pred_mean, pred_var = self(*inputs)
        loss = self.loss_func(pred_mean, target, pred_var)

        if not self.trainer.sanity_checking:
            self.val_metric.update(pred_mean, target)
        return loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        # Unfreeze variance weights after 50 epochs
        if self.current_epoch == WARMUP_EPOCHS:
            for param in self.variance_encoder.parameters():
                param.requires_grad = True
            for param in self.variance_decoder.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Forward pass.'''
        # pylint: disable=arguments-differ,unused-argument
        x = x.squeeze(2)  # remove channel dimension, use time as channel dim instead.
        x_mean, mean_encoder_states = self.mean_encoder(x)
        x_mean = self.mean_decoder(x_mean, mean_encoder_states)
        x_mean = x_mean.unsqueeze(2)  # add channel dimension back
        x_var, var_encoder_states = self.variance_encoder(x)
        x_var = self.variance_decoder(x_var, var_encoder_states)
        x_var = x_var.unsqueeze(2)
        return x_mean, x_var


if __name__ == '__main__':
    test_name = Path(__file__).stem
    model = AleotoricUNet()
    train_dataloaders, val_dataloaders = get_dataset()
    trainer = L.Trainer(
        max_epochs=50,
        logger=get_logger_reruns(test_name, LOGDIR),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback(test_name, CHECKPOINT_DIR)],
    )
    trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
