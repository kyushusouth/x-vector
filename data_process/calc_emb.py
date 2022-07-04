"""
x-vectorを保存するためのファイル
"""

from imageio import save
from omegaconf import DictConfig, OmegaConf
import hydra
import mlflow

import os
from datetime import datetime
from sklearn.manifold import trustworthiness
from tqdm import tqdm
import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from speechbrain.pretrained import EncoderClassifier

from mf_writer import MlflowWriter
from model.model import TDNN
from dataset import x_vec_speechbrain_Dataset, x_vec_speechbrain_trans
from loss_another import GE2ELoss


# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')


def make_data_loader(cfg):
    trans = x_vec_speechbrain_trans(
        cfg=cfg
    )
    dataset = x_vec_speechbrain_Dataset(
        data_root=cfg.train.train_path,
        train=False,
        transform=trans,
        cfg=cfg
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    return data_loader, dataset


def calc_emb(classifier, data_loader, dataset, device):
    iter_cnt = 0
    all_iter = len(data_loader)
    print("iter start")
    embeddings = torch.zeros(len(dataset), 10, 512)
    for batch in tqdm(data_loader):
        # print(f'iter {iter_cnt}/{all_iter}')
        
        wav, speaker_label = batch     
        wav = wav.to(device)
        wav = wav.squeeze(2)
        n_speaker, n_utterance, t = wav.shape
        wav = wav.reshape(n_speaker * n_utterance, t)
        embedding = classifier.encode_batch(wav)
        embedding = embedding.squeeze(1)
        embedding = embedding.reshape(n_speaker, n_utterance, -1)
        embeddings[iter_cnt] = embedding
        
        # if iter_cnt == 10:
        #     break
        iter_cnt += 1
    
    return embeddings



@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    # mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # # mlflowを使ったデータの管理
    # experiment_name = cfg.train.experiment_name
    # writer = MlflowWriter(experiment_name)
    # writer.log_params_from_omegaconf_dict(cfg)

    # 事前学習済みのモデルからx-vectorを抽出する
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")

    # dataloader
    data_loader, dataset = make_data_loader(cfg)

    # embeddingsを計算
    embeddings = calc_emb(
        classifier=classifier,
        data_loader=data_loader,
        dataset=dataset,
        device=device,
    )

    # 保存
    embeddings = embeddings.to('cpu').detach().numpy().copy()
    breakpoint()
    save_path = os.path.join(cfg.train.emb_save_path, current_time)
    os.makedirs(save_path, exist_ok=True)
    np.save(
        os.path.join(save_path, 'emb'),
        embeddings
    )


if __name__ == "__main__":
    main()