from torch.utils.data import Dataset
from dataclasses import dataclass
import os
import pandas as pd
from dataclass_csv import DataclassReader


class UCFCrimeDataset(Dataset):
    def __init__(self, paths):
        """Initialize the dataset with the given list of paths."""
        self.paths = paths
        self.data = []
        for path in paths:
            self.data.append(self.load_data_with_cache(path))

    def __len__(self):
        length = 0
        for path in self.paths:
            with open(path, 'r') as f:
                reader = DataclassReader(f, BBox)
                length += sum(1 for _ in reader)
        return length

    def __getitem__(self, idx):  # TODO: test dataset loading. is it a clip of data per idx or a frame of data?
        return self.data[idx]

    # Cache loaded data to avoid repeated reading
    def load_data_with_cache(csv_path, cache_dir='./cache/'):
        cache_path = os.path.join(cache_dir, f"{os.path.basename(csv_path)}.pkl")
        
        if os.path.exists(cache_path):
            print(f"Loading from cache: {cache_path}")
            return pd.read_pickle(cache_path)
        else:
            print(f"Loading from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            os.makedirs(cache_dir, exist_ok=True)
            df.to_pickle(cache_path)
            return df

    def rank_loss(self, predictions, targets):
        # Compute and return the ranking loss between predictions and targets
        pass


@dataclass
class BBox:
    """Bounding box for detected objects.
        csv dataset read and write reference:
        https://pypi.org/project/dataclass-csv/
    """
    clip: int
    name: str
    frame: int
    person: float
    left: float
    top: float
    width: float
    height: float
    is_anomaly: bool
    anomaly: str