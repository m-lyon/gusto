'''Second UNet experiment with 2D data.
This experiment uses the FFT of the input data as an additional input to the model.
'''

from pathlib import Path

import torch
import lightning as L

from gusto.lib.utils import get_logger_reruns, get_model_checkpoint_callback

from gusto.experiments.unet.lib.utils import LOGDIR, CHECKPOINT_DIR
from gusto.experiments.unet.experiment_1 import UNet, get_dataset


torch.set_float32_matmul_precision('high')


class UNetFFT(UNet):
    '''UNet model.'''

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''Forward pass.

        Args:
            x: input data, shape ``(batch, time, 1, h, w)``
            mask: mask for the data
        '''
        # pylint: disable=arguments-differ,unused-argument
        x = x.squeeze(2)  # remove channel dimension, use time as channel dim instead.
        x_fft: torch.Tensor = torch.fft.fft2(x)  # pylint: disable=not-callable
        # concat as input to the model
        x = torch.cat([x, x_fft.real], dim=1)
        x, encoder_states = self.encoder(x)
        x = self.decoder(x, encoder_states)
        x = x.unsqueeze(2)  # add channel dimension back
        return x


if __name__ == '__main__':
    test_name = Path(__file__).stem
    model = UNetFFT(input_channels=4)
    train_dataloaders, val_dataloaders = get_dataset()
    trainer = L.Trainer(
        max_epochs=50,
        logger=get_logger_reruns(test_name, LOGDIR),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback(test_name, CHECKPOINT_DIR)],
    )
    trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
