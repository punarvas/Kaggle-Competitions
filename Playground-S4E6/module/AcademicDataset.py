from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


class AcademicDataset(Dataset):
    def __init__(self, root: str, transform=None, target_transform=None):
        data = pd.read_csv(root, index_col="id")
        self.X = data.iloc[:, :-1].values
        self.y = data.iloc[:, -1].values

        y_shape = self.y.shape
        self.y = self.y.reshape(-1, 1)
        self.transform = transform
        self.target_transform = target_transform
        self._data_transform()

    def _data_transform(self):
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        x_shape = self.X.shape
        self.X = self.X.reshape(-1, 1, x_shape[1])

    def __getitem__(self, index: int):
        x = self.X[index]
        y = self.y[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y[0])
        return x, y

    def __len__(self):
        return len(self.y)
