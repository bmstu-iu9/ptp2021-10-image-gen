import torch
import torch.nn as nn


class Downsampler(nn.Module):
    def __init__(self, in_size, out_size, first, second, apply_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, first[0], first[1])
        self.conv2 = nn.Conv2d(out_size, out_size, second[0], second[1])
        self.lrelu = nn.LeakyReLU(2)
        self.bnorm = nn.BatchNorm2d(out_size)
        self.apply_batch = apply_batchnorm

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        if self.apply_batch:
            x = self.bnorm(x)
        return x


class DownsamplerLight(nn.Module):
    def __init__(self, in_size, out_size, ker_size, stride, padding=0, apply_batchnorm=False):
        super().__init__()
        self.conv = nn.Conv2d(in_size, out_size, ker_size, stride, padding=padding)
        self.lrelu = nn.LeakyReLU(2)
        self.bnorm = nn.BatchNorm2d(out_size)
        self.b = apply_batchnorm

    def forward(self, x):
        x = self.conv(x)
        x = self.lrelu(x)
        if self.b:
            x = self.bnorm(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, in_size, out_size, first, second, apply_dropout=True, last_act=True):
        super().__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(in_size, out_size, first[0], first[1])
        self.conv_transpose2 = nn.ConvTranspose2d(out_size, out_size, second[0], second[1])
        self.relu = nn.LeakyReLU(2)
        self.dout = nn.Dropout(0.2)
        self.apply_d = apply_dropout
        self.last_act = last_act

    def forward(self, x):
        x = self.conv_transpose1(x)
        x = self.relu(x)
        x = self.conv_transpose2(x)
        if self.last_act:
            x = self.relu(x)
        if self.apply_d:
            x = self.dout(x)
        return x


class UpsamplerLight(nn.Module):
    def __init__(self, in_size, out_size, ker_size, stride, apply_dropout=False, final_act=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_size, out_size, ker_size, stride)
        self.relu = nn.LeakyReLU(2)
        self.dout = nn.Dropout(0.3)
        self.d = apply_dropout
        self.f_act = final_act

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.relu(x)
        if self.f_act:
            x = self.relu(x)
        if self.d:
            x = self.dout(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_stack = nn.ModuleList([
            Downsampler(3, 64, [4, 1], [4, 2], apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            Downsampler(64, 128, [4, 1], [3, 2], apply_batchnorm=False),  # (batch_size, 64, 64, 128)
            Downsampler(128, 256, [4, 1], [4, 2], apply_batchnorm=False ),  # (batch_size, 32, 32, 256)
            Downsampler(256, 512, [3, 1], [2, 2], apply_batchnorm=False),   # (batch_size, 12, 12, 512)
            Downsampler(512, 512, [2, 1], [2, 2], apply_batchnorm=False),  # (batch_size, 5, 5, 512)
            Downsampler(512, 512, [3, 1], [3, 1], apply_batchnorm=False),  # (batch_size, 1, 1, 512)
        ])
        self.up_stack = nn.ModuleList([
            Upsampler(512, 512, [3, 1], [3, 1]),  # (batch_size, 2, 2, 512)
            Upsampler(1024, 512, [3, 2], [2, 1]),  # (batch_size, 16, 16, 512)
            Upsampler(1024, 256, [3, 2], [3, 1], apply_dropout=False),  # (batch_size, 32, 32, 256)
            Upsampler(512, 128, [5, 2], [4, 1], apply_dropout=False),  # (batch_size, 64, 64, 128)
            Upsampler(256, 64, [4, 2], [4, 1], apply_dropout=False),  # (batch_size, 128, 128, 64)
        ])
        self.final = Upsampler(128, 3, [5, 2], [4, 1], last_act=False, apply_dropout=False)  # (batch_size, 256, 256, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x has shape: (batch_size, 256, 256, 3)
        connections = []
        for layer in self.down_stack:
            x = layer(x)
            connections.append(x)
        connections = connections[-2::-1]  # начинаем с маленьких не включая размерности 1х1
        for i in range(len(self.up_stack)):
            x = self.up_stack[i](x)
            x = torch.cat((x, connections[i]), 1)
        x = self.final(x)
        x = self.sigmoid(x)
        return x


class GeneratorLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_stack = nn.ModuleList([
            DownsamplerLight(3, 64, 4, 2, 1),  # (batch_size, 128, 128, 64)
            DownsamplerLight(64, 128, 4, 2, 1),  # (batch_size, 64, 64, 128)
            DownsamplerLight(128, 256, 4, 2, 1),  # (batch_size, 32, 32, 256)
            DownsamplerLight(256, 512, 4, 2, 1),  # (batch_size, 16, 16, 512)
            DownsamplerLight(512, 512, 3, 2, 1),  # (batch_size, 8, 8, 512)
            DownsamplerLight(512, 512, 3, 2, 1, apply_batchnorm=True),  # (batch_size, 4, 4, 512)
            DownsamplerLight(512, 512, 3, 1, apply_batchnorm=True),  # (batch_size, 2, 2, 512)
            DownsamplerLight(512, 512, 2, 1),  # (batch_size, 1, 1, 512)
        ])
        self.up_stack = nn.ModuleList([
            UpsamplerLight(512, 512, 2, 1, apply_dropout=True),  # (batch_size, 2, 2, 512)
            UpsamplerLight(1024, 512, 2, 2, apply_dropout=True),  # (batch_size, 4, 4, 512)
            UpsamplerLight(1024, 512, 2, 2),  # (batch_size, 8, 8, 512)
            UpsamplerLight(1024, 512, 2, 2),  # (batch_size, 16, 16, 512)
            UpsamplerLight(1024, 256, 2, 2),  # (batch_size, 32, 32, 256)
            UpsamplerLight(512, 128, 2, 2),  # (batch_size, 64, 64, 128)
            UpsamplerLight(256, 64, 2, 2),  # (batch_size, 128, 128, 64)
        ])
        self.final = UpsamplerLight(128, 3, 2, 2, final_act=False)  # (batch_size, 256, 256, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x has shape: (batch_size, 256, 256, 3)
        connections = []
        for layer in self.down_stack:
            x = layer(x)
            connections.append(x)
        connections = connections[-2::-1]  # начинаем с маленьких не включая размерности 1х1
        for i in range(len(self.up_stack)):
            x = self.up_stack[i](x)
            x = torch.cat((x, connections[i]), 1)
        x = self.final(x)
        x = self.sigmoid(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(6, 64, 4, 2, 1)
        self.conv_2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv_3 = nn.Conv2d(128, 256, 3, 1)
        self.conv_4 = nn.Conv2d(256, 512, 3, 2, 2)
        self.conv_5 = nn.Conv2d(512, 1, 3, 1)
        self.relu = nn.LeakyReLU(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.relu(x)
        x = self.conv_4(x)
        x = self.relu(x)
        x = self.conv_5(x)
        x = self.sigmoid(x)
        return x


# new generator







