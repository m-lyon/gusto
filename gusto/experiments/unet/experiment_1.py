'''Initial UNet experiment with 2D data.'''

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import lightning as L

from gusto.lib.torch_lightning import LITModel
from gusto.lib.utils import get_logger_reruns, get_model_checkpoint_callback

from gusto.experiments.unet.lib.layers import ConvLayer
from gusto.experiments.unet.lib.dataset import UNetDataset
from gusto.experiments.unet.lib.utils import LOGDIR, CHECKPOINT_DIR


torch.set_float32_matmul_precision('high')


class EncoderLayer(torch.nn.Module):
    '''Encoder layer for UNet.'''

    def __init__(
        self, in_channels: int, out_channels: int, maxpool: bool = True, batch_norm: bool = True
    ):
        super().__init__()
        self.conv1 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            batch_norm=batch_norm,
            relu=True,
        )
        self.conv2 = ConvLayer(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            batch_norm=batch_norm,
            relu=True,
        )
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2) if maxpool else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Forward pass.'''
        x = self.conv1(x)
        x = self.conv2(x)
        before_pool = x
        if self.maxpool is not None:
            x = self.maxpool(x)
        return x, before_pool


class DecoderLayer(torch.nn.Module):
    '''Decoder layer for UNet.'''

    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool = True):
        super().__init__()
        self.upconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            batch_norm=batch_norm,
            relu=True,
        )
        self.conv2 = ConvLayer(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            batch_norm=batch_norm,
            relu=True,
        )

    def forward(self, x: torch.Tensor, before_pool: torch.Tensor) -> torch.Tensor:
        '''Forward pass.'''
        x = self.upconv(x)
        x = torch.cat([x, before_pool], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetEncoder(torch.nn.Module):
    '''UNet Encoder.'''

    def __init__(self, input_channels: int = 2, batch_norm: bool = True):
        super().__init__()
        self.layer1 = EncoderLayer(input_channels, 32, batch_norm=batch_norm)
        self.layer2 = EncoderLayer(32, 64, batch_norm=batch_norm)
        self.layer3 = EncoderLayer(64, 128, batch_norm=batch_norm)
        self.layer4 = EncoderLayer(128, 256, batch_norm=batch_norm)
        self.layer5 = EncoderLayer(256, 512, maxpool=False, batch_norm=batch_norm)

    def forward(self, x: torch.Tensor):
        '''Forward pass.'''
        x, before_pool1 = self.layer1(x)
        x, before_pool2 = self.layer2(x)
        x, before_pool3 = self.layer3(x)
        x, before_pool4 = self.layer4(x)
        x, _ = self.layer5(x)
        return x, (before_pool1, before_pool2, before_pool3, before_pool4)


class UNetDecoder(torch.nn.Module):
    '''UNet Decoder.'''

    def __init__(self, batch_norm: bool = True):
        super().__init__()
        self.layer1 = DecoderLayer(512, 256, batch_norm=batch_norm)
        self.layer2 = DecoderLayer(256, 128, batch_norm=batch_norm)
        self.layer3 = DecoderLayer(128, 64, batch_norm=batch_norm)
        self.layer4 = DecoderLayer(64, 32, batch_norm=batch_norm)
        self.layer5 = ConvLayer(32, 1, kernel_size=1, padding=0, batch_norm=False)

    def forward(self, x: torch.Tensor, encoder_states):
        '''Forward pass.'''
        before_pool1, before_pool2, before_pool3, before_pool4 = encoder_states
        x = self.layer1(x, before_pool4)
        x = self.layer2(x, before_pool3)
        x = self.layer3(x, before_pool2)
        x = self.layer4(x, before_pool1)
        x = self.layer5(x)
        return x


class UNet(LITModel):
    '''UNet model.'''

    def __init__(self, input_channels: int = 2, batch_norm: bool = True):
        super().__init__()
        self.encoder = UNetEncoder(input_channels=input_channels, batch_norm=batch_norm)
        self.decoder = UNetDecoder(batch_norm=batch_norm)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''Forward pass.'''
        # pylint: disable=arguments-differ,unused-argument
        x = x.squeeze(2)  # remove channel dimension, use time as channel dim instead.
        x, encoder_states = self.encoder(x)
        x = self.decoder(x, encoder_states)
        x = x.unsqueeze(2)  # add channel dimension back
        return x


def get_dataset():
    '''Get the training and validation datasets.'''
    train_dataset = UNetDataset(start=0.0, end=0.9, dims=(256, 256), strategy='wrap')
    val_dataset = UNetDataset(start=0.9, end=1.0, dims=(256, 256), strategy='wrap')
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True
    )
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    test_name = Path(__file__).stem
    model = UNet()
    train_dataloaders, val_dataloaders = get_dataset()
    trainer = L.Trainer(
        max_epochs=50,
        logger=get_logger_reruns(test_name, LOGDIR),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback(test_name, CHECKPOINT_DIR)],
    )
    trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
