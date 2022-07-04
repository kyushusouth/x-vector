import os
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm

def main():
    data_root = "/volumes/toshiba_ext/python/x_vector/dataset"
    save_path = "/volumes/toshiba_ext/python/x_vector/dataset/np_files"
    
    data_root_libri = os.path.join(data_root, "librispeech")
    save_path_libri = os.path.join(save_path, "librispeech")
    for curdir, dirs, files in tqdm(os.walk(data_root_libri)):
        for file in files:
            if file.endswith(".flac"):
                if 
                save_path_libri_each = os.path.join(save_path_libri, Path(curdir).parents[1].name, Path(curdir).parents[0].name)
                os.makedirs(save_path_libri_each, exist_ok=True)
                wav, fs = librosa.load(
                    path=os.path.join(curdir, file),
                    sr=16000
                )
                np.save(
                    file=os.path.join(save_path_libri_each, Path(file).stem), 
                    arr=wav
                )
    print("done")


if __name__ == "__main__":
    main()