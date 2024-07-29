import torch


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
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding=0):
        super().__init__()
        self.layer = torch.nn.Conv1d(input_size, output_size, kernel_size, stride, padding)
        self.batchnorm = torch.nn.BatchNorm1d(output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.layer(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        return out


class TransposeConvLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding=0):
        super().__init__()
        self.layer = torch.nn.ConvTranspose1d(input_size, output_size, kernel_size, stride, padding)
        self.batchnorm = torch.nn.BatchNorm1d(output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.layer(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        return out
