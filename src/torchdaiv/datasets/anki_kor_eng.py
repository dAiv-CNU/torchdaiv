from __future__ import annotations

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, extract_archive

from os.path import join, isfile


class AnkiKorEngDataset(Dataset):
    url = "https://www.manythings.org/anki/kor-eng.zip"
    base_dir = join("anki", "kor-eng")

    raw_kor = []
    raw_eng = []

    def __init__(self, root: str, valid=False, test=False, split_rate=(0.5, 0.3, 0.2), transform=None):
        """
        AnkiKorEng Dataset
        :param root: root directory to download and extract the dataset
        :param valid: get validation set
        :param test: get test set
        :param split_rate: split rate for (train, valid, test)
        :param transform: transform function to apply to the data
        """

        super().__init__()

        if sum(split_rate) != 1.0:
            raise ValueError(f"train + valid + test must be equal to 1.0 (currently {sum(split_rate)})")

        if not self.raw_kor or not self.raw_eng:
            self._download(download_root=root)
            self._raw_data, self.info = self._load_data(data_dir=join(root, self.base_dir))

            #self.raw_eng, self.raw_kor = zip(*self._raw_data)
            for eng, kor in zip(*self._raw_data):
                if "자살" in kor or "죽" in kor:
                    continue
                self.raw_eng.append(eng)
                self.raw_kor.append(kor)

        split_rate = [int(len(self.raw_kor) * n) for n in split_rate]

        if valid:
            start = split_rate[0]
            end = split_rate[0] + split_rate[1]
        elif test:
            start = split_rate[0] + split_rate[1]
            end = -1
        else:  # train set
            start = 0
            end = split_rate[0]

        self.kor = self.raw_kor[start:end]
        self.eng = self.raw_eng[start:end]

        print(f"Dataset loaded. {len(self.kor)} samples loaded.")

        self.transform(transform)

    def __getitem__(self, *args, **kwargs):
        return self.kor.__getitem__(*args, **kwargs), self.eng.__getitem__(*args, **kwargs)

    def __len__(self):
        return len(self.kor)

    def transform(self, transform=None):
        if transform is not None:
            try:
                kt, et = transform
            except:
                kt, et = transform, transform

            self.kor = kt(self.kor)
            self.eng = et(self.eng)

    def _download(self, download_root):
        if not isfile(join(download_root, "kor-eng.zip")):
            download_and_extract_archive(self.url, download_root, extract_root=join(download_root, self.base_dir))
        else:
            extract_archive(join(download_root, "kor-eng.zip"), join(download_root, self.base_dir))

        print("Extraction completed.")

    def _load_data(self, data_dir):
        with open(join(data_dir, "kor.txt"), "r", encoding="UTF-8") as data, \
                open(join(data_dir, "_about.txt"), "r", encoding="UTF-8") as about:
            data = [line.strip().split("\t")[:2] for line in data]
            info = about.read()
        return data, info
