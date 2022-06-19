import os
from pathlib import Path


def get_libri_speakers(data_root):    
    items = dict()
    speakers = []
    idx = 0

    # Librispeechの読み込み
    data_root_libri = os.path.join(data_root, "librispeech")
    for curdir, dir, files in os.walk(data_root_libri):
        # break
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
    return speakers


def get_data_ted(data_root):
    speakers = []
    # ted-liumの読み込み
    data_root_ted = os.path.join(data_root, "TEDLIUM_release-3/data/sph")
    for curdir, dir, files in os.walk(data_root_ted):
        if Path(curdir).name == "sph":
            for speaker in files:
                speakers.append(os.path.join(curdir, speaker))
    return speakers


def get_data_libri(speakers):
    audio_path = []
    for speaker in speakers:
        for curdir, dir, files in os.walk(speaker):
            for file in files:
                if file.endswith(".flac"):
                    audio_path.append(os.path.join(curdir, file))
    return audio_path


def main():
    data_root = "/volumes/toshiba_ext"

    speakers_libri = get_libri_speakers(data_root)
    datasets_libri = get_data_libri(speakers_libri)
    
    datasets_ted = get_data_ted(data_root)
    print(len(datasets_libri))
    print(len(datasets_ted))

if __name__ == "__main__":
    main()