import pickle
from typing import Any, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.logger import ColorLogger
from ..utils.progress import ProgressBar
from ..utils.register import Registers

logger = ColorLogger(name="Dataset")


@Registers.dataset_registry.register
class CriteoDataset(Dataset):
    """
    Criteo Dataset for CTR prediction.

    Args:
        data_path (str): Path to the dataset file.
        scaled (str): Scaling method to use. Options: "MinMaxScaler", "StandardScaler", "None".
    """

    def __init__(
        self,
        data_path: str = "../data/criteo_data.parquet",
        scaled: str = "MinMaxScaler",
    ):
        self.X: Optional[List[Tuple]] = []
        self.y: Optional[torch.Tensor] = None
        self.scaler: Optional[Any] = None
        self.scaled: str = scaled
        self.data_path: str = data_path

        self.pd_data: pd.DataFrame = self._load_df_parquet()
        self._preprocess()

    def _load_df_parquet(self) -> pd.DataFrame:
        logger.info(f"Loading criteo data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded criteo data successfully!")
        return df

    def _preprocess(self) -> None:
        logger.info("Preprocessing data...")
        X = self.pd_data.iloc[:, 1:].values  # Features (columns 1-39)
        y = self.pd_data.iloc[:, 0].values  # Labels (column 0)
        dense_x = X[:, :13]
        discrete_x = X[:, 13:]
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
        else:
            logger.info(f"Using no scaler.")

        dense_x = torch.from_numpy(dense_x)
        discrete_x = torch.from_numpy(discrete_x, dtype=torch.long)
        self.y = torch.from_numpy(y, dtype=torch.long)
        with ProgressBar(total=len(dense_x), title="Processing X") as bar:
            for i in range(len(dense_x)):
                self.X.append((dense_x[i], discrete_x[i]))
                bar()

        logger.info("Data preprocessing completed.")

    def __len__(self) -> int:
        return len(self.pd_data)

    def __getitem__(self, idx) -> Tuple[Tuple, torch.Tensor]:
        return self.X[idx], self.y[idx]
