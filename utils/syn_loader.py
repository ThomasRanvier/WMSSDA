"""Dataset setting and data loader for SYN.
Source: https://github.com/liyxi/synthetic-digits/blob/main/synthetic_digits.py
"""

import os
import warnings

import torch
from PIL import Image
import torch.utils.data as data
import wget
import gzip
import shutil


class SynDigits(data.Dataset):
    """Synthetic Digits Dataset.
    """

    resources = [
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_train.pt.gz',
         'd0e99daf379597e57448a89fc37ae5cf'),
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_test.pt.gz',
         '669d94c04d1c91552103e9aded0ee625')
    ]

    training_file = "synth_train.pt"
    test_file = "synth_test.pt"
    classes = [str(i) for i in range(10)]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Init Synthetic Digits dataset."""
        super(SynDigits, self).__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the Synthetic Digits data."""
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            _ = wget.download(url, os.path.join(self.raw_folder, filename))
            with gzip.open(os.path.join(self.raw_folder, filename), 'rb') as f_in:
                with open(os.path.join(self.processed_folder, filename[:-3]), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        print('Download done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")