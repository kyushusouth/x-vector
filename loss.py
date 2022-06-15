#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:58:34 2018

@author: harry
"""

import torch
import torch.nn as nn

from utils import get_centroids, get_cossim, calc_loss

# class SpeechEmbedder(nn.Module):
    
#     def __init__(self):
#         super(SpeechEmbedder, self).__init__()    
#         self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True)
#         for name, param in self.LSTM_stack.named_parameters():
#           if 'bias' in name:
#              nn.init.constant_(param, 0.0)
#           elif 'weight' in name:
#              nn.init.xavier_normal_(param)
#         self.projection = nn.Linear(hp.model.hidden, hp.model.proj)
        
#     def forward(self, x):
#         x, _ = self.LSTM_stack(x.float()) #(batch, frames, n_mels)
#         #only use last frame
#         x = x[:,x.size(1)-1]
#         x = self.projection(x.float())
#         x = x / torch.norm(x, dim=1).unsqueeze(1)
#         return x

class GE2ELoss(nn.Module):
    
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device
        
    def forward(self, embeddings):
        """
        embeddings : (B, n_utterance, C)
        """
        torch.clamp(self.w, 1e-6)
        # 発話に対して平均を取り、発話によらない話者性を表すcentroidsを得る
        centroids = get_centroids(embeddings)   # (B, C)

        # 話者、発話ごとのembeddingsと話者ごとのcentroidsの類似度行列を計算
        cossim = get_cossim(embeddings, centroids)  # (B, n_utterance, B)
        sim_matrix = self.w*cossim.to(self.device) + self.b     # (B, n_utterance, B)

        # 損失
        loss, _ = calc_loss(sim_matrix)
        return loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    batch = 64
    n_utterance = 10
    channels = 256
    embeddings = torch.rand(batch, n_utterance, channels)
    loss_fn = GE2ELoss(device)
    out = loss_fn(embeddings)
    print(out)
    return


if __name__ == "__main__":
    main()