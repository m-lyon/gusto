import torch
import numpy as np


def conv1d_output_size(input_size, padding, kernel_size, stride) -> int:
    '''According to https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html'''
    return np.floor((input_size + 2 * padding - 1 * (kernel_size - 1) - 1) / stride + 1).astype(int)


def calculate_transpose_conv1d_output_length(input_length, padding, kernel_size, stride):
    '''
    Calculate the output length of a 1D transpose convolution (deconvolution) layer.

    Parameters:
    - input_length (int): Length of the input sequence.
    - kernel_size (int): Size of the convolutional kernel.
    - stride (int): Stride of the convolution.
    - padding (int): Amount of zero-padding added to the input.
    - output_padding (int): Additional size added to one side of the output.

    Returns:
    - output_length (int): Calculated length of the output sequence.
    '''
    output_length = stride * (input_length - 1) + kernel_size - 2 * padding
    return output_length


class TimeEmbedding(torch.nn.Module):
    def __init__(self, layer_dims, output_size):
        super().__init__()
        layers = []
        input_dim = layer_dims[0]
        if len(layer_dims) == 1:
            layers.append(LinearLayer(input_dim, output_size))
        else:
            for layer_dim in layer_dims[1:]:
                layers.append(LinearLayer(input_dim, layer_dim))
                input_dim = layer_dim
            layers.append(torch.nn.Linear(input_dim, output_size))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LinearLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = torch.nn.Linear(input_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.layer(x)
        out = self.relu(out)
        return out


class ConvLayer(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        input_len,
        stride=1,
        padding=0,
        apply_batchnorm=True,
    ):
        super().__init__()
        self.layer = torch.nn.Conv1d(input_size, output_size, kernel_size, stride, padding)
        self.batchnorm = torch.nn.BatchNorm1d(output_size) if apply_batchnorm else None
        self.relu = torch.nn.ReLU()
        self.output_len = conv1d_output_size(input_len, padding, kernel_size, stride)

    def forward(self, x):
        out = self.layer(x)
        if self.batchnorm is not None:
            out = self.batchnorm(out)
        out = self.relu(out)
        return out


class MaxPoolLayer(torch.nn.Module):
    def __init__(self, kernel_size, input_len, stride=None, padding=0):
        super().__init__()
        self.layer = torch.nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        self.output_len = conv1d_output_size(
            input_len, padding, kernel_size, kernel_size if stride is None else stride
        )

    def forward(self, x):
        x = self.layer(x)
        # assert (
        #     x.shape[-1] == self.output_len
        # ), f"Output length is {x.shape[-1]}, expected {self.output_len}"
        return x


class UpsampleLayer(torch.nn.Module):
    def __init__(self, input_len, scale_factor):
        super().__init__()
        self.layer = torch.nn.Upsample(scale_factor=scale_factor)
        self.output_len = input_len * scale_factor

    def forward(self, x):
        return self.layer(x)


class FlattenLayer(torch.nn.Module):
    def __init__(self, input_channels, input_len):
        super().__init__()
        self.input_channels = int(input_channels)
        self.input_len = int(input_len)
        self.output_len = int(input_channels * input_len)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        # assert (
        #     x.shape[-1] == self.output_len
        # ), f"Output length is {x.shape[-1]}, expected {self.output_len}"
        return x


class UnflattenLayer(torch.nn.Module):
    def __init__(self, unflatten_size):
        super().__init__()
        self.unflatten_size = unflatten_size[-1]
        self.unflatten = torch.nn.Unflatten(1, unflatten_size)

    def forward(self, x):
        return self.unflatten(x)


class TransposeConvLayer(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        input_len,
        stride=1,
        padding=0,
        apply_batchnorm=True,
    ):
        super().__init__()
        self.layer = torch.nn.ConvTranspose1d(input_size, output_size, kernel_size, stride, padding)
        self.batchnorm = torch.nn.BatchNorm1d(output_size) if apply_batchnorm else None
        self.relu = torch.nn.ReLU()
        self.output_len = calculate_transpose_conv1d_output_length(
            input_len, padding, kernel_size, stride
        )

    def forward(self, x):
        out = self.layer(x)
        if self.batchnorm is not None:
            out = self.batchnorm(out)
        out = self.relu(out)
        return out


class EncoderBlockA(torch.nn.Module):
    def __init__(self, input_dim: int, layer_dims, kernel_sizes, input_len, apply_batchnorm=True):
        super().__init__()
        layers = []
        for layer_dim, kernel_size in zip(layer_dims, kernel_sizes):
            layer = ConvLayer(
                input_dim, layer_dim, kernel_size, input_len, apply_batchnorm=apply_batchnorm
            )
            layers.append(layer)
            input_dim = layer_dim
            input_len = layer.output_len

        self.layers = torch.nn.Sequential(*layers)
        self.output_len = layers[-1].output_len

    def forward(self, x):
        return self.layers(x)


class DecoderBlockA(torch.nn.Module):
    def __init__(self, input_dim: int, layer_dims, kernel_sizes, input_len, apply_batchnorm=True):
        super().__init__()
        layers = []
        for layer_dim, kernel_size in zip(layer_dims, kernel_sizes):
            layer = TransposeConvLayer(
                input_dim, layer_dim, kernel_size, input_len, apply_batchnorm=apply_batchnorm
            )
            input_len = layer.output_len
            layers.append(layer)
            input_dim = layer_dim

        self.layers = torch.nn.Sequential(*layers)
        self.output_len = input_len

    def forward(self, x):
        return self.layers(x)


class CoderBlockB(torch.nn.Module):
    def __init__(
        self, input_size, output_size, input_len, branches=(0, 1, 2, 3), apply_batchnorm=True
    ):
        super().__init__()
        assert all([0 <= b < 4 for b in branches]), "Branches must be in [0, 1, 2, 3]"
        assert (
            output_size % len(branches) == 0
        ), f'Output size {output_size} must be divisible by the number of branches {len(branches)}'
        branch_out_size = output_size // len(branches)
        self.branches = torch.nn.ModuleList()
        if 0 in branches:
            layer = ConvLayer(
                input_size, branch_out_size, 1, input_len, apply_batchnorm=apply_batchnorm
            )
            self.branches.append(layer)
            input_len = layer.output_len
        if 1 in branches:
            layer1 = ConvLayer(
                input_size, branch_out_size, 1, input_len, apply_batchnorm=apply_batchnorm
            )
            input_len = layer1.output_len
            layer2 = ConvLayer(
                branch_out_size,
                branch_out_size,
                7,
                input_len,
                padding=3,
                apply_batchnorm=apply_batchnorm,
            )
            input_len = layer2.output_len
            self.branches.append(torch.nn.Sequential(layer1, layer2))
        if 2 in branches:
            layer1 = ConvLayer(
                input_size, branch_out_size, 1, input_len, apply_batchnorm=apply_batchnorm
            )
            input_len = layer1.output_len
            layer2 = ConvLayer(
                branch_out_size,
                branch_out_size,
                3,
                input_len,
                padding=1,
                apply_batchnorm=apply_batchnorm,
            )
            input_len = layer2.output_len
            layer3 = ConvLayer(
                branch_out_size,
                branch_out_size,
                3,
                input_len,
                padding=1,
                apply_batchnorm=apply_batchnorm,
            )
            input_len = layer3.output_len
            self.branches.append(torch.nn.Sequential(layer1, layer2, layer3))
        if 3 in branches:
            layer1 = MaxPoolLayer(3, input_len, stride=1, padding=1)
            input_len = layer1.output_len
            layer2 = ConvLayer(
                input_size, branch_out_size, 1, input_len, apply_batchnorm=apply_batchnorm
            )
            input_len = layer2.output_len
            self.branches.append(torch.nn.Sequential(layer1, layer2))
        self.output_len = input_len

    def forward(self, x):
        out = [branch(x) for branch in self.branches]
        out = torch.cat(out, 1)
        return out


class TimeDistributed(torch.nn.Module):

    def forward_reshape(self, x):
        '''Reshape input tensor to (samples * timesteps, ...)'''
        samples, timesteps = x.shape[:2]
        return x.view(-1, *x.shape[2:]), samples, timesteps

    def backward_reshape(self, x, samples, timesteps):
        '''Reshape input tensor back to (samples, timesteps, ...)'''
        return x.view(samples, timesteps, *x.shape[1:])


class Encoder(TimeDistributed):
    def __init__(
        self,
        input_size,
        input_len,
        layer_dims_a,
        kernels_a,
        batch_norm_a,
        output_dim_b,
        branches_b,
        batch_norm_b,
        layer_dims_c,
        kernels_c,
        batch_norm_c,
        latent_size,
    ):
        super().__init__()
        self.block_a = EncoderBlockA(input_size, layer_dims_a, kernels_a, input_len, batch_norm_a)
        self.max_pool_a = MaxPoolLayer(3, self.block_a.output_len)
        self.block_b = CoderBlockB(
            layer_dims_a[-1],
            output_dim_b,
            self.max_pool_a.output_len,
            branches_b,
            apply_batchnorm=batch_norm_b,
        )
        self.max_pool_b = MaxPoolLayer(3, self.block_b.output_len)
        self.block_c = EncoderBlockA(
            output_dim_b, layer_dims_c, kernels_c, self.max_pool_b.output_len, batch_norm_c
        )
        self.flatten = FlattenLayer(layer_dims_c[-1], self.block_c.output_len)
        self.last_linear = LinearLayer(self.flatten.output_len, latent_size)

    def forward(self, x):
        '''Runs forward pass'''
        x, samples, timesteps = self.forward_reshape(x)
        x = self.block_a(x)
        x = self.max_pool_a(x)
        x = self.block_b(x)
        x = self.max_pool_b(x)
        x = self.block_c(x)
        x = self.flatten(x)
        x = self.last_linear(x)
        x = self.backward_reshape(x, samples, timesteps)
        return x


class Decoder(TimeDistributed):
    def __init__(
        self,
        latent_size,
        input_size,
        layer_dims_a,
        kernels_a,
        batch_norm_a,
        output_dim_b,
        branches_b,
        batch_norm_b,
        layer_dims_c,
        kernels_c,
        batch_norm_c,
        flatten_input_channels,
        flatten_input_len,
    ):
        super().__init__()
        self.first_layer = LinearLayer(latent_size, flatten_input_channels * flatten_input_len)
        self.unflatten_layer = UnflattenLayer((flatten_input_channels, flatten_input_len))
        self.block_a = DecoderBlockA(
            flatten_input_channels,
            layer_dims_c[::-1][1:] + (output_dim_b,),
            kernels_c[::-1],
            self.unflatten_layer.unflatten_size,
            apply_batchnorm=batch_norm_c,
        )
        self.upsample_layer_a = UpsampleLayer(self.block_a.output_len, 3)
        self.block_b = CoderBlockB(
            output_dim_b,
            layer_dims_a[-1],
            self.upsample_layer_a.output_len,
            branches_b[::-1],
            apply_batchnorm=batch_norm_b,
        )
        self.upsample_layer_b = UpsampleLayer(self.block_b.output_len, 3)
        self.block_c = DecoderBlockA(
            layer_dims_a[-1],
            layer_dims_a[::-1][1:] + (input_size,),
            kernels_a[::-1],
            self.upsample_layer_b.output_len,
            apply_batchnorm=batch_norm_a,
        )

    def forward(self, x):
        '''Runs forward pass'''
        x, samples, timesteps = self.forward_reshape(x)
        x = self.first_layer(x)
        x = self.unflatten_layer(x)
        x = self.block_a(x)
        x = self.upsample_layer_a(x)
        x = self.block_b(x)
        x = self.upsample_layer_b(x)
        x = self.block_c(x)
        x = self.backward_reshape(x, samples, timesteps)
        return x


class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size=1,
        input_len=100,
        layer_dims_a=(32, 64, 128, 128),
        kernels_a=(1, 7, 3, 3),
        batch_norm_a=True,
        output_dim_b=128,
        branches_b=(0, 1, 2, 3),
        batch_norm_b=True,
        layer_dims_c=(64, 32),
        kernels_c=(3, 1),
        batch_norm_c=True,
        time_embedding_layers=(1, 32),
        latent_size=32,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_size=input_size,
            input_len=input_len,
            layer_dims_a=layer_dims_a,
            kernels_a=kernels_a,
            batch_norm_a=batch_norm_a,
            latent_size=latent_size,
            output_dim_b=output_dim_b,
            branches_b=branches_b,
            batch_norm_b=batch_norm_b,
            layer_dims_c=layer_dims_c,
            kernels_c=kernels_c,
            batch_norm_c=batch_norm_c,
        )
        self.time_embedding = TimeEmbedding(time_embedding_layers, latent_size)
        self.time_encoder = torch.nn.LSTM(latent_size, latent_size, 1, batch_first=True)
        self.decoder = Decoder(
            latent_size=latent_size,
            input_size=input_size,
            layer_dims_a=layer_dims_a,
            kernels_a=kernels_a,
            batch_norm_a=batch_norm_a,
            output_dim_b=output_dim_b,
            branches_b=branches_b,
            batch_norm_b=batch_norm_b,
            layer_dims_c=layer_dims_c,
            kernels_c=kernels_c,
            batch_norm_c=batch_norm_c,
            flatten_input_channels=self.encoder.flatten.input_channels,
            flatten_input_len=self.encoder.flatten.input_len,
        )

    def forward(self, x: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor):
        # Encode data
        x = self.encoder(x)
        # Get time embedding for input timestep
        t_in = self.time_embedding(t_in.unsqueeze(-1))
        # Add time embedding to latent representation
        x = x + t_in
        # Apply LSTM
        _, (h, _) = self.time_encoder(x)
        # Add output time embedding
        t_out = self.time_embedding(t_out.unsqueeze(-1))
        # expand final hidden state
        x = h.transpose(0, 1).expand(t_out.size(0), t_out.size(1), -1)
        x = x + t_out
        # Decode
        x = self.decoder(x)
        return x


class AutoEncoderWeightedAverage(AutoEncoder):
    '''AutoEncoder with a weighted average of the last n timepoints as the input to the decoder'''

    def __init__(
        self,
        input_size=1,
        input_len=100,
        layer_dims_a=(32, 64, 128, 128),
        kernels_a=(1, 7, 3, 3),
        batch_norm_a=True,
        output_dim_b=128,
        branches_b=(0, 1, 2, 3),
        batch_norm_b=True,
        layer_dims_c=(64, 32),
        kernels_c=(3, 1),
        batch_norm_c=True,
        time_embedding_layers=(1, 32),
        latent_size=32,
        average_window_size=2,
    ):
        super().__init__(
            input_size=input_size,
            input_len=input_len,
            layer_dims_a=layer_dims_a,
            kernels_a=kernels_a,
            batch_norm_a=batch_norm_a,
            output_dim_b=output_dim_b,
            branches_b=branches_b,
            batch_norm_b=batch_norm_b,
            layer_dims_c=layer_dims_c,
            kernels_c=kernels_c,
            batch_norm_c=batch_norm_c,
            time_embedding_layers=time_embedding_layers,
            latent_size=latent_size,
        )
        self.average_window_size = average_window_size

    def forward(self, x: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor):
        last_x = self.get_average(x, t_in, t_out, self.average_window_size)
        # Encode data
        x = self.encoder(x)
        # Get time embedding for input timestep
        t_in = self.time_embedding(t_in.unsqueeze(-1))
        # Add time embedding to latent representation
        x = x + t_in
        # Apply LSTM
        _, (h, _) = self.time_encoder(x)
        # Add output time embedding
        t_out = self.time_embedding(t_out.unsqueeze(-1))
        # expand final hidden state
        x = h.transpose(0, 1).expand(t_out.size(0), t_out.size(1), -1)
        x = x + t_out
        # Decode
        x = self.decoder(x)
        x = last_x + x
        return x

    def get_average(self, x, t_in, t_out, n):
        '''Get an average of the last n values weighted on the distance from t_out to t_in

        x: data, shape -> (batch, time_in, channels, cpgs)
        t_in: input time, shape -> (batch, time_in)
        t_out: output time, shape -> (batch, time_out)
        n: number of timepoints to average over

        last_x: average of the last n timepoints, shape -> (batch, t_out, channels, cpgs)
        '''
        # calculate weights baed on distance from t_out to t_in, ensuring that the weights sum to 1
        weights = (
            1
            - torch.divide(torch.abs(t_out[:, None, :] - t_in[:, -n:, None]), t_out[:, None, :])[
                ..., None, None
            ]
        )
        # calculate the weighted average
        last_x = torch.sum(x[:, -n:, None, ...] * weights, dim=1)
        return last_x
