import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, latent, features, output_size=64):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(latent, features*8, 4, 1, 0),  # img: 4x4
            self._block(features*8, features*4, 4, 2, 1),  # img: 8x8
            self._block(features*4, features*2, 4, 2, 1),  # img: 16x16
            self._block(features*2, features, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(features, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), # Output: N x channels_img x 64 x 64
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            spectral_norm(nn.Conv2d(3, features, kernel_size=4, stride=2, padding=1)),
            nn.SiLU(inplace=True),
            self._block(features, features*2, 4, 2, 1),
            self._block(features*2, features*4, 4, 2, 1),
            self._block(features*4, features*8, 4, 2, 1),
            spectral_norm(nn.Conv2d(features*8, 1, kernel_size=4, stride=2, padding=0)),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=False)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.disc(x)
