import os
import torch
import torch.nn as nn
from collections import OrderedDict
import librosa
import soundfile

def write_wav(path, audio, sr=8000):
    soundfile.write(path, audio, sr, "PCM_16")

# build UNet model using 1d conv for input data size (1, 32000)
class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet1D, self).__init__()

        features = init_features
        self.encoder1 = UNet1D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet1D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet1D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet1D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet1D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv1d(in_channels, features, kernel_size=3, padding=1)),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2", nn.Conv1d(features, features, kernel_size=3, padding=1)),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        encoder1 = self.encoder1(x)
        pool1 = self.pool1(encoder1)

        encoder2 = self.encoder2(pool1)
        pool2 = self.pool2(encoder2)

        encoder3 = self.encoder3(pool2)
        pool3 = self.pool3(encoder3)

        encoder4 = self.encoder4(pool3)
        pool4 = self.pool4(encoder4)

        bottleneck = self.bottleneck(pool4)

        upconv4 = self.upconv4(bottleneck)
        cat4 = torch.cat((upconv4, encoder4), dim=1)
        decoder4 = self.decoder4(cat4)

        upconv3 = self.upconv3(decoder4)
        cat3 = torch.cat((upconv3, encoder3), dim=1)
        decoder3 = self.decoder3(cat3)

        upconv2 = self.upconv2(decoder3)
        cat2 = torch.cat((upconv2, encoder2), dim=1)
        decoder2 = self.decoder2(cat2)

        upconv1 = self.upconv1(decoder2)
        cat1 = torch.cat((upconv1, encoder1), dim=1)
        decoder1 = self.decoder1(cat1)

        return self.conv(decoder1)