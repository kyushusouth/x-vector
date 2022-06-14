import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class extractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layer = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, 512, kernel_size=5)),
            nn.Tanh(),
            weight_norm(nn.Conv1d(512, 512, kernel_size=3, dilation=2)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(512, 512, kernel_size=3, dilation=3)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(512, 512, kernel_size=1)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(512, 1500, kernel_size=1)),
        )

        self.first_fc = nn.Linear(3000, 256)
        self.second_fc = nn.Linear(256, 6)
        
    def forward(self, x):
        """
        x : (B, C, T)

        return
        x_vec : (B, C)
        out : (B, C)
        """
        # 畳み込み
        out = self.conv_layer(x)

        # stats pooling
        out = torch.cat((out.mean(dim=2), out.std(dim=2)), dim=1)

        x_vec = self.first_fc(out)
        out = F.softmax(self.second_fc(x_vec))
        return x_vec, out


def main():
    net = extractor(80, 5)
    x = torch.rand(5, 80, 300)
    x_vec, out = net(x)
    print(x_vec.shape)
    print(out.shape)


if __name__ == "__main__":
    main()