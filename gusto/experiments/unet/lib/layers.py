'''UNet implementation'''

import torch


class ConvLayer(torch.nn.Module):
    '''Convolutional layer for UNet.'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        stride: int = 1,
        padding: int = 1,
        batch_norm: bool = True,
        relu: bool = True,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias
        )
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        else:
            self.batch_norm = None
        if relu:
            self.relu = torch.nn.ReLU()
        else:
            self.relu = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Runs forward pass'''
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
