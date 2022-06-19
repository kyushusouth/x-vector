import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class extractor(nn.Module):
    def __init__(self, in_channels, n_dim, out_channels):
        super().__init__()

        self.conv_layer = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, 512, kernel_size=5, padding=2)),
            nn.Tanh(),
            weight_norm(nn.Conv1d(512, 512, kernel_size=3, dilation=2, padding=2)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(512, 512, kernel_size=3, dilation=3, padding=3)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(512, 512, kernel_size=1)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(512, 1500, kernel_size=1)),
        )

        self.conv1 = weight_norm(nn.Conv1d(in_channels, 512, kernel_size=5, padding=2))
        self.conv2 = weight_norm(nn.Conv1d(512, 512, kernel_size=3, dilation=2, padding=2))
        self.conv3 = weight_norm(nn.Conv1d(512, 512, kernel_size=3, dilation=3, padding=3))
        self.conv4 = weight_norm(nn.Conv1d(512, 512, kernel_size=1))
        self.conv5 = weight_norm(nn.Conv1d(512, 1500, kernel_size=1))

        self.first_fc = nn.Linear(3000, n_dim)
        self.second_fc = nn.Linear(n_dim, out_channels)
        
    def forward(self, x):
        """
        x : (B=n_speakers, n_utterance, C, T)

        return
        x_vec : (B, n_utterance, C)
        out : (B, n_utterance, C)
        """
        batch_size = x.shape[0]
        n_utterance = x.shape[1]
        x = x.reshape(batch_size * n_utterance, x.shape[2], x.shape[3])

        # 畳み込み
        out = self.conv_layer(x)
        
        # stats pooling
        out = torch.cat((out.mean(dim=2), out.std(dim=2)), dim=1)
    
        x_vec = self.first_fc(out)
        out = F.softmax(self.second_fc(x_vec), dim=1)

        x_vec = x_vec.reshape(batch_size, n_utterance, x_vec.shape[-1])
        out = out.reshape(batch_size, n_utterance, out.shape[-1])
        return x_vec, out


def main():
    net = extractor(40, 6)
    x = torch.rand(64, 10, 40, 300)
    x_vec, out = net(x)
    print(x_vec.shape)
    print(out.shape)


if __name__ == "__main__":
    main()