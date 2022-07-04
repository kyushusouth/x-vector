from sklearn.pipeline import FeatureUnion
import torch
import os
import glob
from pathlib import Path
import random
from tqdm import tqdm

import numpy as np
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import audiomentations
from data_process.feature import wave2mel, wav2world, SoxEffects
import librosa
import matplotlib.pyplot as plt


def get_speakers(data_path):
    """
    話者IDを取得
    """
    speakers = []
    for curdir, dirs, files in os.walk(data_path):
        for dir in dirs:
            if Path(curdir).name == "wav" or Path(curdir).name == "aac":
                if "id" in dir:
                    speakers.append(os.path.join(curdir, dir))
    return speakers


def get_utterance(speaker, n_utterance):
    """
    話者ごとにn_utterance分の音声データを取得
    """
    utterance = []
    for curdir, dirs, files in os.walk(speaker):
        for file in files:
            if Path(file).suffix == ".wav" or Path(file).suffix == ".m4a":
                utterance.append(os.path.join(curdir, file))
    utterance = random.sample(utterance, n_utterance)
    return utterance


class GE2EDataset(Dataset):
    def __init__(self, data_path, name, transform, n_utterance, train):
        super().__init__()

        self.speakers = get_speakers(data_path)
        self.len = len(self.speakers)
        self.n_utterance = n_utterance
        self.transform = transform
        self.name = name

    def __len__(self):
        return self.len

    def get_item(self, speaker, transform):
        utterance = get_utterance(speaker, self.n_utterance)
        if self.transform is not None:
            pass
        else:
            feature = transform(utterance, self.name)
        return feature

    def __getitem__(self, index):
        speaker = self.speakers[index]
        return self.get_item(speaker)


class MySubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.speakers = dataset.speakers
        print(self.__len__())

    def __getitem__(self, idx):
        speaker = self.speakers[self.indices[idx]]
        return self.dataset.get_item(speaker, self.transform)

    def __len__(self):
        return len(self.indices)


class GE2Etrans:
    def __init__(self, musan_path, ir_path, cfg):
        self.cfg = cfg
        self.fs = cfg.model.fs
        self.frame_period = cfg.model.frame_period
        self.f_min = cfg.model.f_min
        self.f_max = cfg.model.f_max
        self.n_mel = cfg.model.n_mel
        self.length = cfg.model.length

        self.add_noise = audiomentations.AddBackgroundNoise(
            sounds_path=os.path.join(musan_path, 'noise'),
            min_snr_in_db=0,
            max_snr_in_db=15,
            p=0.5,
        )
        self.add_music = audiomentations.AddBackgroundNoise(
            sounds_path=os.path.join(musan_path, 'music'),
            min_snr_in_db=5,
            max_snr_in_db=15,
            p=0.5,
        )
        self.add_speech = audiomentations.AddBackgroundNoise(
            sounds_path=os.path.join(musan_path, 'speech'),
            min_snr_in_db=13,
            max_snr_in_db=20,
            p=0.5,
        )
        self.add_ir = audiomentations.ApplyImpulseResponse(
            ir_path=ir_path,
            leave_length_unchanged=True,
            p=0.5,
        )
        self.cut_silence = SoxEffects(
            sample_rate=self.fs,
            sil_threshold=0.3,
            sil_duration=0.1,
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

    def __call__(self, utterance, feature_type):
        features = []
        for i in range(len(utterance)):
            wav, fs = librosa.load(utterance[i])
            wav = wav[None, :]
            assert wav.ndim == 2
            assert wav.shape[-1] != 0

            # save_path = "/home/usr4/r70264c/x_vector/data_check"
            # fig = plt.figure(figsize=(8, 6))
            # ax1 = fig.add_subplot(
            #     211,
            #     title="natural waveform",
            # )
            # ax1.plot(wav.T)

            # 無音区間の切り取り，正規化
            wav = torch.from_numpy(wav)
            wav = self.cut_silence(wav, self.fs)

            # サンプル数の調整
            wav = self.time_adjust(wav)
            wav = wav.to('cpu').detach().numpy().copy()

            # ax2 = fig.add_subplot(
            #     212,
            #     title="processed waveform",
            # )
            # ax2.plot(wav.T)
            # os.makedirs(save_path, exist_ok=True)
            # fig.savefig(os.path.join(save_path, f"{i}.png"))


            # ノイズ，残響付与
            idx = np.random.randint(0, 3)
            if idx == 0:
                wav = self.add_ir(self.add_noise(wav.squeeze(0), self.fs), self.fs)
            elif idx == 1:
                wav = self.add_ir(self.add_music(wav.squeeze(0), self.fs), self.fs)
            else:
                wav = self.add_ir(self.add_speech(wav.squeeze(0), self.fs), self.fs)
            
            if feature_type == "mspec":
                # メルスペクトログラムへ変換
                feature = wave2mel(
                    wave=wav,
                    fs=self.fs,
                    frame_period=self.frame_period,
                    n_mels=self.n_mel,
                    fmin=self.f_min,
                    fmax=self.f_max,
                )
            elif feature_type == "world":
                mcep, clf0, vuv, cap, fbin, t = wav2world(
                    wave=wav,
                    fs=self.fs,
                    frame_period=self.frame_period,
                    comp_mode=self.cfg.model.comp_mode
                )
                feature = np.hstack([mcep, clf0.reshape(-1, 1), vuv.reshape(-1, 1), cap])
            features.append(feature)

        feature = np.stack(features)
        return feature


class GE2Etrans_val(GE2Etrans):
    def __init__(self, musan_path, ir_path, cfg):
        super().__init__(musan_path, ir_path, cfg)

    def __call__(self, utterance, feature_type):
        features = []
        for i in range(len(utterance)):
            wav, fs = librosa.load(utterance[i])
            wav = wav[None, :]
            assert wav.ndim == 2

            # 無音区間の切り取り，正規化
            wav = torch.from_numpy(wav)
            wav = self.cut_silence(wav, self.fs)

            # サンプル数の調整
            wav = self.time_adjust(wav)
            wav = wav.to('cpu').detach().numpy().copy()

            if feature_type == "mspec":
                feature = wave2mel(
                    wave=wav,
                    fs=self.fs,
                    frame_period=self.frame_period,
                    n_mels=self.n_mel,
                    fmin=self.f_min,
                    fmax=self.f_max,
                )
            elif feature_type == "world":
                mcep, clf0, vuv, cap, fbin, t = wav2world(
                    wave=wav,
                    fs=self.fs,
                    frame_period=self.frame_period,
                    comp_mode=self.cfg.model.comp_mode
                )
                feature = np.hstack([mcep, clf0.reshape(-1, 1), vuv.reshape(-1, 1), cap])
            features.append(feature)

        feature = np.stack(features)
        return feature.squeeze(1)
