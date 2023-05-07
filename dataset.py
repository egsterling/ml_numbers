"""
EECS 445 - Introduction to Machine Learning
Winter 2023 - Project 2
Dogs Dataset
    Class wrapper for interfacing with the dataset of dog images
    Usage: python dataset.py
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import config


def get_train_val_test_loaders(batch_size):
    """Return DataLoaders for train, val and test splits.

    Any keyword arguments are forwarded to the DogsDataset constructor.
    """
    tr, va, te, _ = get_train_val_test_datasets()

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)

    print("ldskf", tr_loader)

    return tr_loader, va_loader, te_loader


def get_train_val_test_datasets():
    """Return DogsDatasets and image standardizer.

    Image standardizer should be fit to train data and applied to all splits.
    """
    tr = NumbersDataset("train")
    va = NumbersDataset("val")
    te = NumbersDataset("test")

    # Resize
    # We don't resize images, but you may want to experiment with resizing
    # images to be smaller for the challenge portion. How might this affect
    # your training?
    # tr.X = resize(tr.X)
    # va.X = resize(va.X)
    # te.X = resize(te.X)

    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    te.X = standardizer.transform(te.X)


    return tr, va, te, standardizer


def resize(X):
    """Resize the data partition X to the size specified in the config file.

    Use bicubic interpolation for resizing.

    Returns:
        the resized images as a numpy array.
    """
    image_dim = config("image_dim")
    image_size = (image_dim, image_dim)
    resized = []
    for i in range(X.shape[0]):
        xi = Image.fromarray(X[i]).resize(image_size, resample=2)
        resized.append(xi)
    resized = [np.asarray(im) for im in resized]
    resized = np.array(resized)
    return resized


class ImageStandardizer(object):
    """Standardize a batch of images to mean 0 and variance 1.

    The standardization should be applied separately to each channel.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """

    def __init__(self):
        """Initialize mean and standard deviations to None."""
        super().__init__()
        self.image_mean = None
        self.image_std = None

    def fit(self, X):
        """Calculate per-channel mean and standard deviation from dataset X.
        Hint: be careful of the axis argument. """
        # TODO: Complete this function
        self.image_mean = np.mean(X)
        self.image_std = np.std(X)

    def transform(self, X):
        """Return standardized dataset given dataset X."""
        # TODO: Complete this function

        X_transformed = (X - self.image_mean) / self.image_std
        return X_transformed


def get_file_num(file):
    return int(file.split('/')[3].rstrip('.png'))

class NumbersDataset(Dataset):
    """Dataset class for dog images."""

    def __init__(self, partition):
        """Read in the necessary data from disk.

        For parts 2, 3 and data augmentation, `task` should be "target".
        For source task of part 4, `task` should be "source".

        For data augmentation, `augment` should be True.
        """
        super().__init__()
        if partition not in ["train", "val", "test"]:
            raise ValueError("Partition {} does not exist".format(partition))

        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        self.partition = partition
        # self.task = task
        # Load in all the data we need from disk

        self.metadata = pd.read_csv(config("csv_file"))
        # print(self.metadata[self.metadata.file.apply(get_file_num) % 3 == 0])
        # print(self.metadata)
        self.X, self.y = self._load_data()

    def __len__(self):
        """Return size of dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return (image, label) pair at index `idx` of dataset."""
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

    def _load_data(self):
        """Load a single data partition from file."""
        print("loading %s..." % self.partition)
        partition_dict = {"train": 0, "val": 1, "test": 2}

        df = self.metadata[
            self.metadata.file.apply(get_file_num) % 48 == partition_dict[self.partition]
        ]

        path = config("image_path")

        X, y = [], []
        for _, row in df.iterrows():
            label = row["label"]
            image = imread(os.path.join(path, row["file"]))
            X.append(image)
            y.append(label)
        return np.expand_dims(np.array(X), axis=1), np.array(y)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    tr, va, te, standardizer = get_train_val_test_datasets()
    print("Train:\t", len(tr.X))
    print("Val:\t", len(va.X))
    print("Test:\t", len(te.X))
    print("Mean:", standardizer.image_mean)
    print("Std: ", standardizer.image_std)


