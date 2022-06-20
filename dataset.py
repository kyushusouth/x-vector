"""
データセットは(話者数、発話数、チャンネル数、時間)になる
話者数がバッチサイズとして指定されればいい
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
import audiomentations


def get_speakers(data_root):    
    items = dict()
    speakers = []
    idx = 0

    # Librispeechの読み込み
    data_root_libri = os.path.join(data_root, "librispeech")
    for curdir, dir, files in os.walk(data_root_libri):
        break
        # for file in files:
            # if file.endswith(".flac"):
            #     audio_path = os.path.join(curdir, file)
            #     label = Path(audio_path).parents[1].stem
            #     if os.path.isfile(audio_path):
            #             items[idx] = [audio_path, label]
            #             idx += 1
        if Path(curdir).name == "train-clean-100":
            for speaker in dir:
                speakers.append(os.path.join(curdir, speaker))
        elif Path(curdir).name == "train-clean-360":
            for speaker in dir:
                speakers.append(os.path.join(curdir, speaker))
        elif Path(curdir).name == "train-other-500":
            for speaker in dir:
                speakers.append(os.path.join(curdir, speaker))

    # ted-liumの読み込み
    data_root_ted = os.path.join(data_root, "TEDLIUM_release-3/data/sph")
    for curdir, dir, files in os.walk(data_root_ted):
        if Path(curdir).name == "sph":
            for speaker in files:
                speakers.append(os.path.join(curdir, speaker))
    return speakers


def get_dataset_libri(speaker, train, cfg):
    """
    librispeechのデータ取得用関数
    """
    audio_path = []

    for curdir, dir, files in os.walk(speaker):
        for file in files:
            if file.endswith(".flac"):
                audio_path.append(os.path.join(curdir, file))
                # label = Path(audio_path).parents[1].stem
    # if train:
    #     audio_path = random.sample(audio_path, cfg.train.n_utterance)
    # else:
    #     audio_path = random.sample(audio_path, 1)
    audio_path = random.sample(audio_path, cfg.train.n_utterance)
    return audio_path


def calc_mean_std(speakers, cfg):
    """
    speakers : 全話者のパスのリスト

    ここは一回求めた値を何回も使ったほうが良さそう
    """
    try:
        npz_key = np.load(f'{cfg.train.mean_std_path}/{cfg.model.name}.npz')
        mean = torch.from_numpy(npz_key['mean'])
        std = torch.from_numpy(npz_key['std'])
    except:
        audio_path = []
        mean = 0
        std = 0
        # 全音声データをロード
        for speaker in speakers:
            # ted-lium
            if speaker.endswith(".sph"):
                audio_path.append(speaker)

            # librispeech
            else:
                for curdir, dir, files in os.walk(speaker):
                    for file in files:
                        if file.endswith(".flac"):
                            audio_path.append(os.path.join(curdir, file))

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

            # wav = wav.to('cpu').detach().numpy().copy()
            # np.save()

        # データ数で平均
        mean /= len(audio_path)
        std /= len(audio_path)
        np.savez(
            f'{cfg.train.mean_std_path}/{cfg.model.name}',
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

        print(f"data : {self.len}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        return
        mel : (B=num_speakers, num_utterance, C, T)
        """
        # [audio_path, label] = self.items[idx]
        speaker = self.speakers[idx]

        speaker_label = Path(speaker).stem

        # ted-lium
        if speaker.endswith(".sph"):
            audio_path = speaker
            wav, fs = torchaudio.backend.soundfile_backend.load(audio_path)

            # fsを16kHzに変換
            wav = F.resample(
                waveform=wav,
                orig_freq=fs,
                new_freq=self.cfg.model.fs,
            )

            utterance_len = self.cfg.model.length    # 1発話の長さ
            n_utterance = self.cfg.train.n_utterance     # 総発話数

            # 適当に3秒分の発話を総発話数分だけ取得するためのindex
            idx = torch.randint(0, wav.shape[1] - utterance_len * n_utterance, (n_utterance,))
            
            mel_save = []
            for i in idx:
                wav_utterance = wav[:, i:i + utterance_len]
                mel_save.append(self.trans(wav_utterance, self.mean, self.std, self.train))
            
        # librispeech
        else:
            audio_path = get_dataset_libri(speaker, self.train, self.cfg)
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

        if self.train:
            assert len(mel_save) == self.cfg.train.n_utterance

        # 0初期化した行列に計算結果を代入。リストからtensorへ変換。
        mel = torch.zeros(self.cfg.train.n_utterance, mel_save[0].shape[0], mel_save[0].shape[1])
        for i in range(len(mel_save)):
            mel[i] = mel_save[i]

        return mel, speaker_label


class x_vec_speechbrain_Dataset(Dataset):
    def __init__(self, data_root, train, transform, cfg):
        super().__init__()
        self.train = train
        self.trans = transform
        self.cfg = cfg

        self.speakers = get_speakers(data_root)
        self.len = len(self.speakers)
        self.mean, self.std = calc_mean_std(self.speakers, self.cfg)

        print(f"data : {self.len}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        return 
        """
        speaker = self.speakers[idx]
        speaker_label = Path(speaker).stem

        # ted-lium
        if speaker.endswith(".sph"):
            audio_path = speaker
            wav, fs = torchaudio.backend.soundfile_backend.load(audio_path)

            # fsを16kHzに変換
            wav = F.resample(
                waveform=wav,
                orig_freq=fs,
                new_freq=self.cfg.model.fs,
            )

            utterance_len = self.cfg.model.length    # 1発話の長さ
            n_utterance = self.cfg.train.n_utterance     # 総発話数

            # 適当に3秒分の発話を総発話数分だけ取得するためのindex
            idx = torch.randint(0, wav.shape[1] - utterance_len * n_utterance, (n_utterance,))
            wav_save = []
            for i in idx:
                wav_utterance = wav[:, i:i + utterance_len]
                wav_save.append(self.trans(wav_utterance, self.mean,self.std, self.train))
        
        # librispeech
        else:
            audio_path = get_dataset_libri(speaker, self.train, self.cfg)
            wav_save = []
            for i in range(len(audio_path)):
                wav, fs = torchaudio.backend.soundfile_backend.load(audio_path[i])

                # fsを16kHzに変換
                wav = F.resample(
                    waveform=wav,
                    orig_freq=fs,
                    new_freq=self.cfg.model.fs,
                )

                wav_save.append(self.trans(wav))

        assert len(wav_save) == self.cfg.train.n_utterance
        wav = torch.zeros(self.cfg.train.n_utterance, wav_save[0].shape[0], wav_save[0].shape[1])
        for i in range(len(wav_save)):
            wav[i] = wav_save[i]

        return wav, speaker_label
        


class x_vec_trans:
    def __init__(self, cfg):
        self.cfg = cfg
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
        self.add_noise = audiomentations.AddBackgroundNoise(
            sounds_path=os.path.join(cfg.train.musan_path, 'noise'),
            min_snr_in_db=0,
            max_snr_in_db=15,
            p=0.5,
        )
        self.add_music = audiomentations.AddBackgroundNoise(
            sounds_path=os.path.join(cfg.train.musan_path, 'music'),
            min_snr_in_db=5,
            max_snr_in_db=15,
            p=0.5,
        )
        self.add_speech = audiomentations.AddBackgroundNoise(
            sounds_path=os.path.join(cfg.train.musan_path, 'speech'),
            min_snr_in_db=13,
            max_snr_in_db=20,
            p=0.5,
        )
        self.add_ir = audiomentations.ApplyImpulseResponse(
            ir_path=cfg.train.ir_path,
            leave_length_unchanged=True,
            p=0.5,
        )
        # self.trans = audiomentations.Compose([
        #     audiomentations.Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5),
        #     audiomentations.AddBackgroundNoise(sounds_path=cfg.train.noise_path, min_snr_in_db=3, max_snr_in_db=20, p=0.5),
        #     audiomentations.ApplyImpulseResponse(ir_path=cfg.train.ir_path, leave_length_unchanged=True, p=0.5),
        # ])

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
        wav = self.time_adjust(wav)

        if train:
            # audiomentationによるdata augmentationのため，一度tensorからnumpyへ変換
            # torchaudio.save(os.path.join("/volumes/toshiba_ext/check_augmentation", "before.wav"), wav, self.cfg.model.fs)    # trans前データ
            wav = wav.to('cpu').detach().numpy().copy()

            # x_vectorの論文を参考に付加するものによってsn比を変えたので，どれかひとつを適用するようにしている
            idx = torch.randint(0, 2, (1,))
            if idx == 0:
                wav = self.add_ir(self.add_noise(wav.squeeze(0), self.cfg.model.fs), self.cfg.model.fs)
            elif idx == 1:
                wav = self.add_ir(self.add_music(wav.squeeze(0), self.cfg.model.fs), self.cfg.model.fs)
            else:
                wav = self.add_ir(self.add_speech(wav.squeeze(0), self.cfg.model.fs), self.cfg.model.fs)

            # wav = self.trans(wav.squeeze(0), self.cfg.model.fs)
            wav = torch.from_numpy(wav).unsqueeze(0)
            # torchaudio.save(os.path.join("/volumes/toshiba_ext/check_augmentation", "after.wav"), wav, self.cfg.model.fs)     # trans後データ
            # breakpoint()
            
        # 正規化
        wav = self.normalize(wav, mean, std)

        # 音声をメルスペクトログラムに変換
        mel = self.wav2mel(wav)     
        return mel.squeeze(0)


class x_vec_speechbrain_trans(x_vec_trans):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        
    def __call__(self, wav, mean, std, train):
        wav = self.time_adjust(wav)
        wav = self.normalize(wav, mean, std)
        return wav