from cProfile import label
import hydra
import torch
import os
from pathlib import Path

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


def get_datasets(data_root):    
    items = dict()
    idx = 0
    for curdir, dir, files in os.walk(data_root):
        for file in files:
            if file.endswith(".flac"):
                audio_path = os.path.join(curdir, file)
                label = Path(audio_path).parents[1].stem
                if os.path.isfile(audio_path):
                        items[idx] = [audio_path, label]
                        idx += 1
    return items


class x_vec_Dataset(Dataset):
    def __init__(self, data_root, train, transform, cfg):
        super().__init__()
        self.train = train
        self.trans = transform
        self.cfg = cfg
        self.items = get_datasets(data_root)
        self.len = len(self.items)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        [audio_path, label] = self.items[idx]
        wav, fs = torchaudio.backend.soundfile_backend.load(audio_path)

        # fsを16kHzに変換
        wav = F.resample(
            waveform=wav,
            orig_freq=fs,
            new_freq=self.cfg.model.fs,
        )

        # メルスペクトログラムに変換
        mel = self.trans(wav)
        return mel, label


class x_vec_trans:
    def __init__(self, cfg):
        self.length = cfg.model.length
        self.wav2mel = T.MelSpectrogram(
            sample_rate=cfg.model.fs,
            n_fft=cfg.model.n_fft,
            hop_length=cfg.model.hop_length,
            win_length=cfg.model.win_length,
            f_min=cfg.model.f_min,
            f_max=cfg.model.f_max,
            n_mels=cfg.model.n_mels,
        )

    def time_adjust(self, wav):
        if wav.shape[1] <= self.length:
            wav_padded = torch.zeros(1, self.length)
            wav_padded[:, :wav.shape[1]] = wav
            for i in range(self.length - wav.shape[1]):
                # self.lengthを満たすまで、原音声のサンプルを繰り返す
                wav_padded[:, wav.shape[1] + i] = wav[:, i % wav.shape[1]]

            wav = wav_padded
        
        elif wav.shape[1] > self.length:
            # 適当にself.lengthだけとる
            idx = torch.randint(0, int(wav.shape[1]) - self.length, (1,))
            wav = wav[:, idx:idx + self.length]

        assert wav.shape[1] == self.length, "time adjust error"
        return wav

    def __call__(self, wav):
        wav = self.time_adjust(wav)
        mel = self.wav2mel(wav)
        return mel.squeeze(0)

@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    trans = x_vec_trans(
        cfg=cfg
    )
    dataset = x_vec_Dataset(
        data_root=cfg.model.train_path,
        train=True,
        transform=trans,
        cfg=cfg
    )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    for batch in train_loader:
        mel, label = batch
        print(mel.shape)
        print(len(label))


if __name__ == "__main__":
    main()

    