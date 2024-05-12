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
            self.general_text.append(d['RawText'])
            self.general_label.append(int(d['GeneralPolarity']))
            for ae in d['Aspects']:
                self.sentiment_text.append(ae['SentimentText'])
                self.sentiment_label.append(int(ae['SentimentPolarity']))

        self.data = self.sentiment_text if sentiment else self.general_text
        self.label = self.sentiment_label if sentiment else self.general_label

        if transform is not None:
            self.data = transform(self.data)

        if target_transform is not None:
            self.label = target_transform(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

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
