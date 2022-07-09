"""
学習したモデルを読み込み,TSNEを書く
"""

from argparse import ArgumentParser
from pathlib import Path
from unicodedata import name
from warnings import filterwarnings

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio
from librosa.util import find_files
from sklearn.manifold import TSNE
from tqdm import tqdm
import hydra

from torch.utils.data import DataLoader, Dataset
from model.model import TDNN
from dataset_remake import GE2EDataset, MySubset, GE2Etrans
import os
import random
import numpy as np

random.seed(7)


def get_speakers(data_path):
    speakers = []
    for curdir, dirs, files in os.walk(data_path):
        if Path(curdir).name == "test":
            for dir in dirs:
                speakers.append(os.path.join(curdir, dir))
    assert speakers is not None
    speakers = random.sample(speakers, 10)  # 適当に10人取得
    return speakers


def get_utterance(speaker):
    utterance = []
    for curdir, dirs, files in os.walk(speaker):
        for file in files:
            if Path(file).suffix == ".wav" or Path(file).suffix == ".m4a":
                utterance.append(os.path.join(curdir, file))
    utterance = random.sample(utterance, len(utterance))
    # utterance = random.shuffle(utterance)
    assert utterance is not None
    return utterance
                

class TestDataset(Dataset):
    def __init__(self, data_path, name, transform, n_utterance):
        super().__init__()
        self.name = name
        self.speakers = get_speakers(data_path)
        self.len = len(self.speakers)
        self.transform = transform
        print(f"num_speaker = {self.len}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        speaker = self.speakers[idx]
        utterance = get_utterance(speaker)
        feature = self.transform(utterance, self.name)
        return feature, Path(speaker).name


def make_test_loader(data_path, cfg, device):
    trans_test = GE2Etrans(
        musan_path=None,
        ir_path=None,
        cfg=cfg,
        calc_eer=False,
    )
    dataset = TestDataset(
        data_path=data_path,    
        name=cfg.model.name,
        transform=trans_test,
        n_utterance=cfg.train.n_utterance,
    )
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=1,   
        shuffle=False,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = TDNN(
        in_channels=cfg.model.in_channels,
        n_dim=cfg.model.n_dim,
        out_channels=cfg.model.out_channels,
    ).to(device)

    model_path = Path("/home/usr4/r70264c/x_vector/checkpoint/2022:07:05_18-59-17/mspec_20.ckpt")
    model_key = torch.load(model_path)
    model.load_state_dict(model_key["model"])

    data_path = cfg.test.data_path  # 設定
    test_loader = make_test_loader(data_path, cfg, device)

    x_vecs = []
    speaker_labels = []
    iter_count = 0
    for batch in test_loader:
        feature, speaker = batch
        feature = feature.to(device)

        with torch.no_grad():
            x_vec, out = model(feature)

        x_vec = x_vec.squeeze(0)
        x_vec = x_vec.to('cpu').detach().numpy().copy()
        print(x_vec.shape)
        x_vecs.append(x_vec)
        speaker_labels.append(speaker)

    assert len(x_vecs) == len(speaker_labels)
    x_vec = np.concatenate(x_vecs, axis=0)
    print(f"x_vec = {x_vec.shape}")
    print(f"speaker_labels = {speaker_labels}")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    x_vec_trans = tsne.fit_transform(x_vec)
    print(f"x_vec_trans = {x_vec_trans.shape}")
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    for i in range(len(speaker_labels)):
        ax.scatter(
            x=x_vec_trans[len(speaker_labels) * i : len(speaker_labels) * i + len(speaker_labels), 0],
            y=x_vec_trans[len(speaker_labels) * i : len(speaker_labels) * i + len(speaker_labels), 1],
            label=str(speaker_labels[i]),
            alpha=0.5,
        )
    save_path = Path(cfg.test.save_path)
    save_path = save_path / model_path.parents[0].name / cfg.model.name / model_path.stem
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path / "tsne.png")


if __name__ == "__main__":
    main()