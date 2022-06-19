
import hydra

import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    embeddings = np.load(os.path.join(cfg.train.emb_save_path, 'emb.npy'))
    n_speaker, n_utterance, _ = embeddings.shape
    print(embeddings.shape)

    tsne = TSNE(n_components=2, learning_rate='auto')
    embeddings_tsne = np.zeros((n_speaker, n_utterance, 2))

    for i in tqdm(range(embeddings.shape[0])):
        embedding = tsne.fit_transform(embeddings[i])
        embeddings_tsne[i] = embedding

    np.save(
        os.path.join(cfg.train.emb_save_path, 'emb_tsne'),
        embeddings_tsne
    )

    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(1, 1, 1)
    # for i in range(len(embeddings_tsne)):
    #     ax.scatter(embeddings_tsne[i][:, 0], embeddings_tsne[i][:, 1])

    # plt.colorbar()
    # plt.savefig(os.path.join(cfg.train.emb_save_path, 'librispeech_tsne.png'))
    
@hydra.main(config_name="config", config_path="conf")
def plot_tsne(cfg):
    embeddings = np.load(os.path.join(cfg.train.emb_save_path, 'emb.npy'))
    print(embeddings.shape)
    n_speaker, n_utterance, n_dim = embeddings.shape

    cos = torch.nn.CosineSimilarity(dim=1)
    embeddings = torch.from_numpy(embeddings)

    print("### similarity per utterance ###")
    for i in range(n_utterance - 1):
        similarity = cos(embeddings[:, i, :], embeddings[:, i+1, :])
        print(torch.mean(similarity))

    print("\n### similarity per speaker ###")
    for i in range(n_speaker - 1):
        similarity = cos(embeddings[i, :, :], embeddings[i+1, :, :])
        print(torch.mean(similarity))
        breakpoint()

    

    


if __name__ == "__main__":
    # main()
    plot_tsne()