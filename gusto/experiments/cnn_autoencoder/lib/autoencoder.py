import torch

from gusto.lib.layers import ConvLayer, LinearLayer, TransposeConvLayer


class TimeDistributed(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):

        # Squash samples and timesteps into a single axis
        x_reshape = x.view(-1, *x.shape[2:])  # (samples * timesteps, ...)
        y = self.module(x_reshape)

        # reshape Y
        y = y.view(*x.shape[:2], *y.shape[1:])  # (samples, timesteps, ...)
        return y


class TimeEmbedding(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            LinearLayer(input_size, output_size),
            torch.nn.Linear(output_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        bracnh_out_size = output_size // 4
        self.branch1 = ConvLayer(input_size, bracnh_out_size, 1)
        self.branch2 = torch.nn.Sequential(
            ConvLayer(input_size, bracnh_out_size, 1),
            ConvLayer(bracnh_out_size, bracnh_out_size, 7, padding=3),
        )
        self.branch3 = torch.nn.Sequential(
            ConvLayer(input_size, bracnh_out_size, 1),
            ConvLayer(bracnh_out_size, bracnh_out_size, 3, padding=1),
            ConvLayer(bracnh_out_size, bracnh_out_size, 3, padding=1),
        )
        self.branch4 = torch.nn.Sequential(
            torch.nn.MaxPool1d(3, stride=1, padding=1), ConvLayer(input_size, bracnh_out_size, 1)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], 1)
        return out


class Encoder(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        layers = torch.nn.Sequential(
            ConvLayer(input_size, 64, 7),
            ConvLayer(64, 128, 1),
            ConvLayer(128, 128, 3),
            torch.nn.MaxPool1d(3),
            EncoderBlock(128, 128),
            torch.nn.MaxPool1d(3),
            ConvLayer(128, 64, 3),
            ConvLayer(64, 32, 1),
            torch.nn.Flatten(),
            LinearLayer(8 * 32, output_size),
        )
        self.layers = TimeDistributed(layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        layers = torch.nn.Sequential(
            LinearLayer(input_size, 8 * 32),
            torch.nn.Unflatten(1, (32, 8)),
            TransposeConvLayer(32, 64, 1),
            TransposeConvLayer(64, 128, 3),
            torch.nn.Upsample(scale_factor=3),
            EncoderBlock(128, 128),
            torch.nn.Upsample(scale_factor=3),
            TransposeConvLayer(128, 64, 3),
            TransposeConvLayer(64, 32, 7),
            TransposeConvLayer(32, 16, 3),
            TransposeConvLayer(16, 1, 1),
        )
        self.layers = TimeDistributed(layers)

    def forward(self, x):
        return self.layers(x)


class AutoEncoder(torch.nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.encoder = Encoder(1, latent_size)
        self.time_embedding = TimeEmbedding(1, latent_size)
        self.time_encoder = torch.nn.LSTM(latent_size, latent_size, 1, batch_first=True)
        self.decoder = Decoder(latent_size)

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
    def __init__(self, latent_size, average_window_size=2):
        super().__init__(latent_size)
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
