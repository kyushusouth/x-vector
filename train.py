"""
使用方法
1. conf/trainのパス変更
    data_path以外を変更してください

2. wandb.loginの変更

3. train.pyの実行
    学習したパラメータはcheckpointディレクトリに保存されていきます
    最終的な結果はresult/trainに保存されます

4. visualize.pyの実行
    ここで学習したパラメータをロードし,tsneを保存できるようにしています
    tnse.pngという名前でresult/generateに保存されると思います
"""


from omegaconf import DictConfig, OmegaConf
import hydra
import mlflow
from datetime import datetime
import torchaudio
import wandb

import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import random
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

from mf_writer import MlflowWriter
from model.model import TDNN
from dataset_remake import GE2EDataset, MySubset, GE2Etrans
from loss_another import GE2ELoss

# wandbへのログイン
wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "random": random.getstate(),
                "np_random": np.random.get_state(), 
                "torch": torch.get_rng_state(),
                "torch_random": torch.random.get_rng_state(),
                'cuda_random' : torch.cuda.get_rng_state(),
				'epoch': epoch}, ckpt_path)


def make_train_val_loader(data_path, cfg, device):
    # 学習用，検証用それぞれに対してtransformを作成
    trans_train = GE2Etrans(
        musan_path=cfg.train.musan_path,
        ir_path=cfg.train.ir_path,
        cfg=cfg,
        calc_eer=False,
    )
    trans_val = GE2Etrans(
        musan_path=None,
        ir_path=None,
        cfg=cfg,
        calc_eer=False,
    )
    trans_eer = GE2Etrans(
        musan_path=None, 
        ir_path=None,
        cfg=cfg,
        calc_eer=True,
    )
    # trans_val = GE2Etrans_val(
    #     musan_path=cfg.train.musan_path,
    #     ir_path=cfg.train.ir_path,
    #     cfg=cfg,
    # )

    # 元となるデータセットの作成(transformは必ずNoneでお願いします)
    dataset = GE2EDataset(
        data_path=data_path,    
        name=cfg.model.name,
        transform=None,
        n_utterance=cfg.train.n_utterance,
    )

    # 学習用と検証用にデータセットを分割
    n_samples = len(dataset)
    train_size = int(n_samples * 0.95)
    indices = np.arange(n_samples)
    train_dataset = MySubset(
        dataset=dataset,
        indices=indices[:train_size],
        transform=trans_train,
    )
    val_dataset = MySubset(
        dataset=dataset,
        indices=indices[train_size:],
        transform=trans_val,
    )

    # EER計算用のデータセットも作成
    train_dataset_eer = MySubset(
        dataset=dataset,
        indices=indices[:train_size],
        transform=trans_eer,
    )
    val_dataset_eer = MySubset(
        dataset=dataset,
        indices=indices[train_size:],
        transform=trans_eer,
    )

    # それぞれのdata loaderを作成
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=False,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    train_loader_eer = DataLoader(
        dataset=train_dataset_eer,
        batch_size=cfg.train.batch_size,   
        shuffle=False,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    val_loader_eer = DataLoader(
        dataset=val_dataset_eer,
        batch_size=cfg.train.batch_size,   
        shuffle=False,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return train_loader, val_loader, train_loader_eer, val_loader_eer


def train_one_epoch(model, train_loader, optimizer, loss_f_soft, loss_f_cont, device, cfg):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    model.train()
    for batch in train_loader:
        iter_cnt += 1

        feature = batch
        feature = feature.to(device)

        batch_size = feature.shape[0]
        data_cnt += batch_size

        # 出力
        x_vec, out = model(feature)

        # 損失
        softmax_loss = loss_f_soft(x_vec)
        # contrast_loss = loss_f_cont(x_vec)
        # loss = softmax_loss + contrast_loss
        loss = softmax_loss
        
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()

        epoch_loss += loss.item()
        
        # wandb.log({"softmax_loss": softmax_loss.item()})
        # wandb.log({"contrast_loss": contrast_loss.item()})
        wandb.log({"train_iter_loss": loss.item()})

        if cfg.train.debug:
            if iter_cnt > 5:
                break

    # 平均
    epoch_loss /= iter_cnt
    return epoch_loss


def calc_val_loss(model, val_loader, loss_f_soft, loss_f_cont, device, cfg):
    val_loss = 0
    data_cnt = 0
    iter_cnt = 0
    print("--- validation ---")
    model.eval()
    for batch in val_loader:
        iter_cnt += 1

        feature = batch
        feature = feature.to(device)

        batch_size = feature.shape[0]
        data_cnt += batch_size

        with torch.no_grad():
            x_vec, out = model(feature)
        
        # loss = loss_f_soft(x_vec) + loss_f_cont(x_vec)
        loss = loss_f_soft(x_vec)

        val_loss += loss.item()

        wandb.log({"val_iter_loss": loss.item()})
        # writer.log_metric("val_iter_loss", loss.item())

        if cfg.train.debug:
            if iter_cnt > 5:
                break
                
    val_loss /= iter_cnt
    return val_loss


def _calc_eer(model, data_loader, device):
    """
    EERを求めるために考えましたが,そもそも話者ごとのコサイン類似度のラベルが分からないのでどうしたらいいのか検討中です
    """
    model.eval()
    scores = []
    labels = []
    for batch in data_loader:
        feature = batch
        feature = feature.to(device)
        with torch.no_grad():
            x_vec, out = model(feature)
        
        # 同じ話者の2発話なのでラベルは1
        score = F.cosine_similarity(x_vec[:, 0, :], x_vec[:, 1, :], dim=1)
        label = torch.ones_like(score)
        scores.append(score.item())
        labels.append(label.item())

        # 異なる話者の2発話の場合も計算するため，一度シャッフル
        B, U, _ = x_vec.shape
        speaker = torch.arange(B).unsqueeze(1)
        pairs = []
        for i in range(B):
            for j in range(U):
                pairs.append(x_vec[i, j, :], speaker[i])
        pairs = random.sample(pairs, len(pairs))

        # 異なる話者の場合はlabelが-1にならないので違いました…
        for i in range(0, len(pairs), 2):
            x_vec1, speaker1 = pairs[i]
            x_vec2, speaker2 = pairs[i + 1]
            score = F.cosine_similarity(x_vec1, x_vec2, dim=0)
            if speaker1 - speaker2 == 0:
                label = 1
            else:
                label = -1
            scores.append(score)
            labels.append(label)

    scores = np.array(scores)
    labels = np.ones_like(scores)   # 同じ話者の2発話なのでコサイン類似度は1になるはず
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def calc_eer(model, train_loader_eer, val_loader_eer, device, cfg):
    print("--- calc EER ---")
    train_eer = _calc_eer(model, train_loader_eer, device)
    val_eer = _calc_eer(model, val_loader_eer, device)
    wandb.log({"train_EER": train_eer})


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    # wandbでhydraを使うための設定
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )

    if cfg.train.debug:
        print('--- debug ---')

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    print(f"num_cpu = {os.cpu_count()}")
    torch.backends.cudnn.benchmark = True

    # パラメータの保存先
    save_path = os.path.join(cfg.train.save_path, current_time)
    os.makedirs(save_path, exist_ok=True)

    # checkpoint
    ckpt_path = os.path.join(cfg.train.ckpt_path, current_time)
    os.makedirs(ckpt_path, exist_ok=True)

    # dataloader作成
    train_loader, val_loader, train_loader_eer, val_loader_eer = make_train_val_loader(
        data_path=cfg.train.data_path, 
        cfg=cfg, 
        device=device
    )

    # 損失関数
    loss_f_soft = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='softmax') # for softmax loss
    loss_f_cont = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='contrast') # for contrast loss

    # training
    with wandb.init(**cfg.wandb.setup, config=wandb_cfg) as run:
        # ネットワーク
        model = TDNN(
            in_channels=cfg.model.in_channels,
            n_dim=cfg.model.n_dim,
            out_channels=cfg.model.out_channels,
        ).to(device)

        # 最適化手法
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,
            amsgrad=True,
        )

        # scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=cfg.train.max_epoch // 4,
            gamma=cfg.train.lr_decay_rate
        )

        wandb.watch(model, **cfg.wandb.watch)

        if cfg.train.debug:
            max_epoch = cfg.train.debug_max_epoch
        else:
            max_epoch = cfg.train.max_epoch

        model.train()
        for epoch in range(max_epoch):
            print(f"##### {epoch} #####")
            epoch_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f_soft=loss_f_soft, 
                loss_f_cont=loss_f_cont, 
                device=device, 
                cfg=cfg,
            )
            print(f"epoch_loss = {epoch_loss}")
            # writer.log_metric("epoch_loss", epoch_loss)

            if epoch % cfg.train.val_step == 0:
                val_loss = calc_val_loss(
                    model=model, 
                    val_loader=val_loader, 
                    loss_f_soft=loss_f_soft, 
                    loss_f_cont=loss_f_cont, 
                    device=device, 
                    cfg=cfg,
                )
                print(f"val_loss = {val_loss}")
                # writer.log_metric("val_loss", val_loss)
            
            scheduler.step()

            if epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{epoch}.ckpt")
                )

        
        # モデルパラメータの保存
        torch.save(model.state_dict(), os.path.join(save_path, f"model_{cfg.model.name}.pth"))
        # artifact = wandb.Artifact('model', type='model')
        # artifact.add_file(os.path.join(save_path, "model.pth"))
        # wandb.log_artifact(artifact)
        
    wandb.finish()


import librosa
def test():
    data_path = "/home/usr4/r70264c/dataset/sv/wav_files/dev/aac/id00490/3GGcAPc3Xiw/00001.m4a"
    # data_path = "/home/usr4/r70264c/dataset/sv/wav_files/wav/id10058/CzPPdvz1aY4/00001.wav"
    fs = 16000
    wav, fs = librosa.load(data_path, sr=fs)
    wav = wav[None, :]
    print(wav.shape)
    print(fs)

    wav, fs = torchaudio.load(data_path)
    print(wav.shape)
    print(fs)


if __name__ == "__main__":
    main()
    # test()