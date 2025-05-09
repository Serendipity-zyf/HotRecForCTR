import torch
import numpy as np
import pandas as pd

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

from utils.logger import ColorLogger
from utils.progress import ProgressBar
from utils.register import Registers
from torch.utils.data import Dataset
from torch.utils.data import random_split

logger = ColorLogger(name="Dataset")


@Registers.dataset_registry.register
class CriteoDataset(Dataset):
    """
    Criteo Dataset for CTR prediction.

    Args:
        data_path (str): Path to the dataset file.
        scaled (str): Scaling method to use. Options: "MinMaxScaler", "StandardScaler", "None".
        split (bool): Whether to split dataset into train and validation.
        val_ratio (float): Validation set ratio, default is 0.2.
        seed (int): Random seed for splitting, default is 42.
    """

    def __init__(
        self,
        data_path: str,
        scaled: str = "MinMaxScaler",
        split: bool = True,
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.name: str = "CriteoDataset"
        self.X: Optional[List[Tuple[torch.Tensor, ...]]] = []
        self.y: Optional[List[torch.Tensor]] = None
        self.scaler: Optional[Any] = None
        self.scaled: str = scaled
        self.data_path: str = data_path
        self.feature_dims: Optional[List[int]] = None
        self.dense_feature_dims: Optional[int] = None
        self.interact_feature_nums: Optional[int] = None

        self.pd_data: pd.DataFrame = self._load_df_parquet()
        self._get_features()
        self._preprocess()

        if split:
            self.train_dataset, self.val_dataset = self.train_val_split(val_ratio, seed)
        else:
            self.train_dataset = self
            self.val_dataset = None

    def _load_df_parquet(self) -> pd.DataFrame:
        logger.info(f"Loading criteo data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded criteo data successfully!")
        return df

    def _get_features(self) -> None:
        """Get the number of different features in the dataset."""
        self.dense_feature_dims = len([col for col in self.pd_data.columns if "C_" in col])
        self.interact_feature_nums = len([col for col in self.pd_data.columns if "x_I" in col])
        self.feature_dims = self.pd_data.iloc[:, self.dense_feature_dims + 1 :].nunique().tolist()

    def _preprocess(self) -> None:
        logger.info("Preprocessing data...")
        # Extract features and labels
        X = self.pd_data.iloc[:, 1:].values  # Features (columns 1-39)
        y = self.pd_data.iloc[:, 0].values  # Labels (column 0)
        dense_x = X[:, : self.dense_feature_dims]
        discrete_x = X[:, self.dense_feature_dims :]
        # Scaling
        if self.scaled == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler

            logger.info(f"Using MinMaxScaler.")
            self.scaler = MinMaxScaler()
            dense_x = self.scaler.fit_transform(dense_x)
        elif self.scaled == "StandardScaler":
            from sklearn.preprocessing import StandardScaler

            logger.info(f"Using StandardScaler.")
            self.scaler = StandardScaler()
            dense_x = self.scaler.fit_transform(dense_x)
        elif self.scaled == "LogScaler":
            logger.info(f"Using LogScaler scaling.")
            mask = dense_x > 2
            dense_x[mask] = np.log(dense_x[mask]) ** 2
        else:
            logger.info(f"Using no scaler.")

        # Use .float() to ensure it is of float32 type
        dense_x = torch.from_numpy(dense_x).float()
        discrete_x = torch.from_numpy(discrete_x).long()
        self.y = torch.from_numpy(y).long()
        with ProgressBar(total=len(dense_x), title="Processing X") as bar:
            for i in range(len(dense_x)):
                self.X.append((dense_x[i], discrete_x[i]))
                bar()

        logger.info("Data preprocessing completed.")
        # show describe
        logger.info(f"Data description: \n{self.pd_data.describe()}")
        logger.info(f"Data sample: \n{self.pd_data.head()}")
        logger.info(f"Dataset dense feature dimension: {self.dense_feature_dims}")
        logger.info(f"Dataset discrete feature dimensions: {len(self.feature_dims)}")

    def __len__(self) -> int:
        return len(self.pd_data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        return self.X[idx][0], self.X[idx][1], self.y[idx]

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42) -> Tuple[Dataset, Dataset]:
        """Split dataset into training and validation sets.

        Args:
            val_ratio: Ratio of validation set size to total dataset size.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (training dataset, validation dataset)
        """
        total_size = len(self)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            self, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )

        logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
        return train_dataset, val_dataset

    @staticmethod
    def get_dataloader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader for the dataset.

        Args:
            dataset: Dataset to create DataLoader for
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes

        Returns:
            DataLoader instance
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
