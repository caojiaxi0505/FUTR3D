import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ForegroundPredictNet(nn.Module):
    def __init__(self, in_channels=256, out_channels=1):
        super(ForegroundPredictNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=1, padding=0), nn.Sigmoid()
        )
        self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
