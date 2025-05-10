import torch
import pickle
import numpy as np

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Dict

from .tools import UIDVocab
from utils.logger import ColorLogger
from utils.progress import ProgressBar
from utils.register import Registers
from torch.utils.data import Dataset

logger = ColorLogger(name="Dataset")


@Registers.dataset_registry.register
class AmazonDataset(Dataset):
    """
    Amazon Electronic Dataset for CTR prediction.

    Args:
        data_path (str): Path to the dataset file.
        user_vocab_path (str): Path to the user vocabulary file.
        item_vocab_path (str): Path to the item vocabulary file.
        category_vocab_path (str): Path to the category vocabulary file.
        item_info_path (str): Path to the item information file.
        uid_vocab (UIDVocab): User vocabulary object.
        item_vocab (UIDVocab): Item vocabulary object.
        category_vocab (UIDVocab): Category vocabulary object.
        scaled (str): Scaling method to use. Options: "MinMaxScaler", "StandardScaler", "LogScaler".
        data_type (str): Type of data to load. Options: "train", "test", "valid".
    """

    def __init__(
        self,
        data_path: str,
        user_vocab_path: str,
        item_vocab_path: str,
        category_vocab_path: str,
        item_info_path: str,
        uid_vocab: Optional[UIDVocab] = None,
        item_vocab: Optional[UIDVocab] = None,
        category_vocab: Optional[UIDVocab] = None,
        scaled: str = "LogScaler",
        data_type: str = "train",
    ):
        self.name: str = "AmazonDataset"
        self.X: Optional[List[Tuple[torch.Tensor, ...]]] = []
        self.y: Optional[List[torch.Tensor]] = []
        self.dense: Optional[torch.Tensor] = []
        self.scaler: Optional[Any] = None

        self.data_path: str = data_path
        self.user_vocab_path: str = user_vocab_path
        self.item_vocab_path: str = item_vocab_path
        self.category_vocab_path: str = category_vocab_path
        self.item_info_path: str = item_info_path
        self.data_type: str = data_type
        self.scaled: str = scaled

        if uid_vocab is not None and item_vocab is not None:
            self.uid_vocab = uid_vocab
            self.item_vocab = item_vocab
            self.category_vocab = category_vocab
        else:
            self.uid_vocab, self.item_vocab, self.category_vocab = self._load_vocab()
        self.txt_data: List[str] = self._load_data()
        self.item_info: Dict = self._load_pkl(item_info_path)
        self._preprocess()

    def _load_data(self) -> List[str]:
        logger.info(f"Loading {self.data_type} data from {self.data_path}")
        with open(self.data_path, "r") as f:
            txt_data = f.readlines()[:1000]
        return txt_data

    def _load_vocab(self) -> Tuple[UIDVocab, ...]:
        """Load uid_vocab from the dataset."""
        logger.info(f"Loading {self.data_type} vocab from {self.user_vocab_path} and {self.item_vocab_path}")
        return (
            UIDVocab.load(self.user_vocab_path),
            UIDVocab.load(self.item_vocab_path),
            UIDVocab.load(self.category_vocab_path),
        )

    def _load_pkl(self, path: str) -> Any:
        """Load a pickle file."""
        logger.info(f"Loading {self.data_type} item info from {self.item_info_path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def _preprocess(self) -> None:
        logger.info("Start Preprocessing data")
        dense_feature_list = []
        with ProgressBar(total=len(self.txt_data), title="Constructing") as pbar:
            for _, line in enumerate(self.txt_data):
                uid, iid, seq, label = line.strip().split("\t")
                uid_idx = torch.tensor(self.uid_vocab[uid]).long()
                iid_idx = torch.tensor(self.item_vocab[iid]).long()
                seq_idx = self.item_vocab[seq.split("<sep>")]
                valid_len = torch.tensor(len(seq_idx)).long()

                seq_item_info = self.category_vocab[
                    [self.item_info[seq_id]["category"] for seq_id in seq_idx]
                ]
                seq_cat_idx = torch.tensor(seq_item_info).long() if seq_idx else torch.tensor([0]).long()
                seq_idx = torch.tensor(seq_idx).long() if seq_idx else torch.tensor([0]).long()

                label = torch.tensor(int(label)).long()
                item_info = self.item_info[iid]
                cate_idx = torch.tensor(self.category_vocab[item_info["category"]]).long()

                self.X.append((uid_idx, iid_idx, cate_idx, seq_idx, seq_cat_idx, valid_len))
                self.y.append(label)

                dense_feature = [item_info["average_rating"], item_info["rating_number"]]
                dense_feature_list.append(dense_feature)
                pbar()

        dense_feature_list = np.array(dense_feature_list, dtype=np.float32)
        if self.scaled == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler

            logger.info(f"Using MinMaxScaler.")
            self.scaler = MinMaxScaler()
            dense_feature_list = self.scaler.fit_transform(dense_feature_list)
        elif self.scaled == "StandardScaler":
            from sklearn.preprocessing import StandardScaler

            logger.info(f"Using StandardScaler.")
            self.scaler = StandardScaler()
            dense_feature_list = self.scaler.fit_transform(dense_feature_list)
        elif self.scaled == "LogScaler":
            logger.info(f"Using LogScaler scaling.")
            mask = dense_feature_list > 2
            dense_feature_list[mask] = np.log(dense_feature_list[mask]) ** 2

        self.dense = torch.from_numpy(dense_feature_list).float()
        logger.info("Preprocessing data finished!")

    def __len__(self) -> int:
        return len(self.txt_data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        return self.X[idx], self.dense[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        """Custom collate function to handle variable-length sequences."""
        PADING_VALUE = 0
        uid_idx, iid_idx, cate_idx, seq_idx, seq_cat_idx, valid_len = zip(*[item[0] for item in batch])
        dense = torch.stack([item[1] for item in batch], dim=0)
        label = torch.stack([item[2] for item in batch], dim=0)

        uid_idx = torch.stack(uid_idx, dim=0)
        iid_idx = torch.stack(iid_idx, dim=0)
        cate_idx = torch.stack(cate_idx, dim=0)
        seq_idx = torch.nn.utils.rnn.pad_sequence(seq_idx, batch_first=True, padding_value=PADING_VALUE)
        seq_cat_idx = torch.nn.utils.rnn.pad_sequence(
            seq_cat_idx, batch_first=True, padding_value=PADING_VALUE
        )
        valid_len = torch.stack(valid_len, dim=0)
        max_len = torch.max(valid_len).item()
        mask = torch.arange(max_len).expand(len(valid_len), max_len) < valid_len

        return (uid_idx, iid_idx, cate_idx, seq_idx, seq_cat_idx, mask), dense, label

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
            collate_fn=AmazonDataset.collate_fn,
        )
