from torch.utils.data import Dataset
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import torch


class DummyDataset(Dataset):
    def __init__(self, n_features, train=True):
        self.train = train
        seed = 0
        x, y = make_regression(
            50000,
            n_features,
            random_state=seed,
            n_informative=n_features,
        )

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=seed
        )

        if train:
            self.data, self.targets = x_train, y_train
        else:
            self.data, self.targets = x_test, y_test

    def __len__(self):
        if self.train:
            return self.data.shape[0]
        else:
            return self.data.shape[0]

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float)
        target = torch.tensor(self.targets[idx], dtype=torch.float)

        return data, target
