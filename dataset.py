"""
データセットは(話者数、発話数、チャンネル数、時間)になる
話者数がバッチサイズとして指定されればいい

今は全ての話者からランダムに発話を取得するようになっており、ちょっと適当
学習データとテストデータを別ディレクトリに分けておかないと学習時に使用したデータをテスト時も使ってしまうので、後々やっとく

ただ、江崎さんはテストを鏑木研のデータにのみ行っているので、librispeechとかでテストデータを分けとかなくてもいいかも
"""
import hydra
import torch
import os
from pathlib import Path
import random
from tqdm import tqdm

import numpy as np
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


def get_speakers(data_root):    
    items = dict()
    speakers = dict()
    speakers = []
    idx = 0
    for curdir, dir, files in os.walk(data_root):
        # for file in files:
            # if file.endswith(".flac"):
            #     audio_path = os.path.join(curdir, file)
            #     label = Path(audio_path).parents[1].stem
            #     if os.path.isfile(audio_path):
            #             items[idx] = [audio_path, label]
            #             idx += 1
        if Path(curdir).name == "train-clean-100":
            for speaker in dir:
                speakers.append(f"{curdir}/{speaker}")
        # elif Path(curdir).name == "train-clean-360":
        #     for speaker in dir:
        #         speakers.append(f"{curdir}/{speaker}")
        # elif Path(curdir).name == "train-other-500":
        #     for speaker in dir:
        #         speakers.append(f"{curdir}/{speaker}")
    return speakers


def get_dataset(speaker, train):
    audio_path = []
    for curdir, dir, files in os.walk(speaker):
        for file in files:
            if file.endswith(".flac"):
                audio_path.append(os.path.join(curdir, file))
                # label = Path(audio_path).parents[1].stem
    if train:
        audio_path = random.sample(audio_path, 10)
    else:
        audio_path = random.sample(audio_path, 1)
    return audio_path


# def calc_sampling_n(audio_path):
#     """
#     平均と標準偏差を求めるためのサンプリング数の計算
#     mは信頼度95%の場合は1.96、信頼度99%の場合は2.58
#     """
#     data_n_all = len(audio_path)
#     m = 1.96    # 信頼度95%
#     eps = 0.05
#     population_ratio = 0.5

#     a = (eps / m)**2
#     b = (data_n_all - 1) / (population_ratio * (1 - population_ratio))

#     sampling_n = data_n_all / (a * b + 1)
#     return int(sampling_n)


def calc_mean_std(speakers, cfg):
    """
    speakers : 全話者のパスのリスト

    ここは一回求めた値を何回も使ったほうが良さそう
    """
    try:
        npz_key = np.load(f'{cfg.model.mean_std_path}/{cfg.model.name}.npz')
        mean = torch.from_numpy(npz_key['mean'])
        std = torch.from_numpy(npz_key['std'])
    except:
        audio_path = []
        mean = 0
        std = 0
        # 全音声データをロード
        for speaker in speakers:
            for curdir, dir, files in os.walk(speaker):
                for file in files:
                    if file.endswith(".flac"):
                        audio_path.append(os.path.join(curdir, file))
        
        # 平均、標準偏差を求めるサンプル数を計算
        # sampling_n = calc_sampling_n(audio_path)
        # audio_path = random.sample(audio_path, sampling_n)

        # 平均、標準偏差を計算
        print("--- calc mean and std ---")
        for i in tqdm(range(len(audio_path))):
            wav, fs = torchaudio.backend.soundfile_backend.load(audio_path[i])
            
            # fsを16kHzに変換
            wav = F.resample(
                waveform=wav,
                orig_freq=fs,
                new_freq=cfg.model.fs,
            )

            # 時間方向に平均、標準偏差を計算
            mean += torch.mean(wav, dim=1)
            std += torch.std(wav, dim=1)
        # データ数で平均
        mean /= len(audio_path)
        std /= len(audio_path)
        np.savez(
            f'{cfg.model.mean_std_path}/{cfg.model.name}',
            mean=mean,
            std=std
        )
    return mean, std


class x_vec_Dataset(Dataset):
    def __init__(self, data_root, train, transform, cfg):
        super().__init__()
        self.train = train
        self.trans = transform
        self.cfg = cfg

        self.speakers = get_speakers(data_root)
        self.len = len(self.speakers)

        self.mean, self.std = calc_mean_std(self.speakers, self.cfg)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        return
        mel : (B=num_speakers, num_utterance, C, T)
        """
        # [audio_path, label] = self.items[idx]
        speaker = self.speakers[idx]

        # ある話者から音声データまでのパスを取得
        audio_path = get_dataset(speaker, self.train)

        mel_save = []
        for i in range(len(audio_path)):
            wav, fs = torchaudio.backend.soundfile_backend.load(audio_path[i])

            # fsを16kHzに変換
            wav = F.resample(
                waveform=wav,
                orig_freq=fs,
                new_freq=self.cfg.model.fs,
            )

            # メルスペクトログラムに変換
            mel_save.append(self.trans(wav, self.mean, self.std, self.train))

        # 0初期化した行列に計算結果を代入。リストからtensorへ変換。
        mel = torch.zeros(len(audio_path), mel_save[0].shape[0], mel_save[0].shape[1])
        for i in range(len(audio_path)):
            mel[i] = mel_save[i]

        return mel


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

    def normalize(self, wav, mean, std):
        wav = (wav - mean.unsqueeze(0)) / std.unsqueeze(0)
        return wav

    def __call__(self, wav, mean, std, train):
        # 学習時はサンプル数調整
        if train:
            wav = self.time_adjust(wav)

        # 正規化
        wav = self.normalize(wav, mean, std)

        # 音声をメルスペクトログラムに変換
        mel = self.wav2mel(wav)     
        return mel.squeeze(0)

@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    trans = x_vec_trans(
        cfg=cfg
    )
    # dataset
    dataset_train = x_vec_Dataset(
        data_root=cfg.model.train_path,
        train=True,
        transform=trans,
        cfg=cfg
    )
    dataset_test = x_vec_Dataset(
        data_root=cfg.model.train_path,
        train=False,
        transform=trans,
        cfg=cfg
    )
    # loader
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    for batch in train_loader:
        mel = batch
        print(mel.shape)

    


if __name__ == "__main__":
    main()

    