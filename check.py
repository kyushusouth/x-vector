import numpy as np
import librosa

def main():
    path = '/home/usr4/r70264c/dataset/sv/wav_files/dev/aac/id00898/AVknLO_78nM/00019.m4a'
    wav, fs = librosa.load(path, sr=None)
    print(wav.shape)
    print(fs)

if __name__ == "__main__":
    main()