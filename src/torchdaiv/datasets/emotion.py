from __future__ import annotations

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from tqdm.notebook import tqdm

from json import loads

from os.path import join, isdir
from os import listdir
import shutil

import zipfile


def extract_zip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


class EmotionDataset(Dataset):
    url = "https://github.com/dAiv-CNU/torchdaiv/raw/dataset/AI_HUB/%EC%86%8D%EC%84%B1%EA%B8%B0%EB%B0%98%20%EA%B0%90%EC%A0%95%EB%B6%84%EC%84%9D%20%EB%8D%B0%EC%9D%B4%ED%84%B0/download.tar"
    base_dir = join("147.속성기반_감정분석_데이터", "01-1.정식개방데이터")
    meta = dict(
        train=(join("Training", "02.라벨링데이터"), "train"),
        valid=(join("Validation", "02.라벨링데이터"), "valid")
    )
    work_dir = "extracted"

    class Emotion(int):
        POSITIVE = 1
        NEUTRAL = 0
        NEGATIVE = -1

        @staticmethod
        def __new__(cls, *args, **kwargs):
            args = list(args)
            value = int(args[0])
            args[0] = 1 if value >= 1 else -1 if value <= -1 else 0
            return super().__new__(cls, *args, **kwargs)

        def __add__(self, other):
            return EmotionDataset.Emotion(super().__add__(other))

        def __sub__(self, other):
            return EmotionDataset.Emotion(super().__sub__(other))

        def __mul__(self, other):
            return EmotionDataset.Emotion(super().__mul__(other))

        def __truediv__(self, other):
            return EmotionDataset.Emotion(super().__truediv__(other))

        def toint(self) -> int:
            return 0 + self

        def __repr__(self):
            if self >= self.POSITIVE:
                return "긍정"
            elif self == self.NEUTRAL:
                return "중도"
            else:
                return "부정"

    Emotion.POSITIVE = Emotion(1)
    Emotion.NEUTRAL = Emotion(0)
    Emotion.NEGATIVE = Emotion(-1)

    def __init__(self, root: str, download=False, train=True, sentiment=False, transform=None, target_transform=None):
        super(EmotionDataset, self).__init__()

        if download:
            self._download(download_root=root)

        if train:
            data_dir = join(root, self.work_dir, self.meta['train'][1])
        else:
            data_dir = join(root, self.work_dir, self.meta['valid'][1])

        self.jsons = self._load_data(data_dir)

        self.general_text = []
        self.general_label = []
        self.sentiment_text = []
        self.sentiment_label = []

        for d in self.jsons:
            if 'RawText' in d and 'GeneralPolarity' in d:
                self.general_text.append(d['RawText'])
                self.general_label.append(self.Emotion(d['GeneralPolarity']))
            if 'Aspects' in d:
                for ae in d['Aspects']:
                    if 'SentimentText' in ae and 'SentimentPolarity' in ae:
                        self.sentiment_text.append(ae['SentimentText'])
                        self.sentiment_label.append(self.Emotion(ae['SentimentPolarity']))

        self.data = self.sentiment_text if sentiment else self.general_text
        self.label = self.sentiment_label if sentiment else self.general_label

        self.transform(transform, target_transform)

    def __getitem__(self, item: int | tuple[int, int]):
        if isinstance(item, int) or isinstance(item, tuple):
            return self.data[item], self.label[item]
        else:
            raise ValueError("Unsupported item type. Only int or tuple[int, int] is supported.")

    def __len__(self):
        return len(self.data)

    def transform(self, transform=None, target_transform=None):
        if transform is not None:
            self.data = transform(self.data)

        if target_transform is not None:
            self.label = target_transform(self.label)

    def _download(self, download_root):
        download_and_extract_archive(self.url, download_root)

        print("Extracting files recursively...")

        if isdir(self.work_dir):
            shutil.rmtree(self.work_dir)

        trainset = join(download_root, self.base_dir, self.meta['train'][0])
        with tqdm(total=100, mininterval=0.001) as pbar:
            for path in listdir(trainset):
                extract_zip_file(join(trainset, path), join(download_root, self.work_dir, self.meta['train'][1]))
                pbar.update(4)

        validset = join(download_root, self.base_dir, self.meta['valid'][0])
        with tqdm(total=100, mininterval=0.001) as pbar:
            for path in listdir(validset):
                extract_zip_file(join(validset, path), join(download_root, self.work_dir, self.meta['valid'][1]))
                pbar.update(4)

        print("Extraction completed.")

    def _load_data(self, data_dir):
        data = []
        for path in listdir(data_dir):
            with open(join(data_dir, path), "r", encoding="UTF-8") as f:
                data.extend(loads(f.read()))
        return data


if __name__ == "__main__":
    dataset = EmotionDataset(root="./data", download=True)
    print(dataset)
