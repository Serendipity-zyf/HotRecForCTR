import os
import pickle
import json
import collections

from typing import List, Tuple, Optional, Union, Counter


class UIDVocab:
    """UID vocabulary, used for user ID and item ID indexing in CTR prediction"""

    def __init__(
        self,
        uids: Optional[Union[List[str], Tuple[str, ...]]] = None,
        min_freq: int = 0,
    ):
        if uids is None:
            uids = []

        # Sorted by frequency of occurrence
        counter = self._count_uids(uids)
        self._uid_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # The index of the unknown UID is 1
        self.idx_to_uid = ["<pad>", "<unk>"]
        self.uid_to_idx = {uid: idx for idx, uid in enumerate(self.idx_to_uid)}

        # Add UIDs with a frequency greater than or equal to min_freq
        for uid, freq in self._uid_freqs:
            if freq < min_freq:
                continue
            if uid not in self.uid_to_idx:
                self.idx_to_uid.append(uid)
                self.uid_to_idx[uid] = len(self.idx_to_uid) - 1

    def __len__(self) -> int:
        """Return the number of UIDs in the vocabulary"""
        return len(self.idx_to_uid)

    def __getitem__(self, uids: Union[str, List[str], Tuple[str, ...]]) -> Union[int, List[int]]:
        """Convert UID to index"""
        if not isinstance(uids, (list, tuple)):
            return self.uid_to_idx.get(uids, self.unk)
        return [self.__getitem__(uid) for uid in uids]

    def to_uids(self, indices: Union[int, List[int], Tuple[int, ...]]) -> Union[str, List[str]]:
        """Convert index to UID"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_uid[indices]
        return [self.idx_to_uid[index] for index in indices]

    def add(self, uid: str) -> int:
        """Add a new UID to the vocabulary"""
        if uid not in self.uid_to_idx:
            self.idx_to_uid.append(uid)
            self.uid_to_idx[uid] = len(self.idx_to_uid) - 1
            return self.uid_to_idx[uid]
        return self.uid_to_idx[uid]

    @property
    def pad(self) -> int:
        """Return the index of the padding UID"""
        return 0

    @property
    def unk(self) -> int:
        """Return the index of the unknown UID"""
        return 1

    @property
    def uid_freqs(self) -> List[Tuple[str, int]]:
        """Return UID and its frequency"""
        return self._uid_freqs

    def _count_uids(self, uids) -> Counter:
        """Frequency of statistical UID"""
        if len(uids) == 0 or isinstance(uids[0], list):
            uids = [uid for line in uids for uid in line]
        return collections.Counter(uids)

    def save(self, path: str, method: str = "json") -> None:
        """Save vocabulary to file"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        if method == "pickle":
            # Use pickle to save the complete pair
            with open(path, "wb") as f:
                pickle.dump(self, f)
        elif method == "json":
            # Use JSON to save, including frequency information
            data = {"idx_to_uid": self.idx_to_uid, "uid_freqs": self._uid_freqs}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif method == "txt":
            # Save using a text file, one UID per line
            with open(path, "w", encoding="utf-8") as f:
                f.write("<unk>\n")
                for uid, _ in self._uid_freqs:
                    if uid in self.uid_to_idx:
                        f.write(f"{uid}\n")
        else:
            raise ValueError(f"Unsupported save method: {method}")

    @classmethod
    def load(cls, path: str, method: str = "json") -> "UIDVocab":
        """
        Load vocabulary from file Parameters
        """
        if method == "pickle":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif method == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            vocab = cls()
            vocab.idx_to_uid = data["idx_to_uid"]
            vocab._uid_freqs = data["uid_freqs"]
            vocab.uid_to_idx = {uid: idx for idx, uid in enumerate(vocab.idx_to_uid)}
            return vocab
        elif method == "txt":
            uids = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    uid = line.strip()
                    if uid:
                        uids.append(uid)
            # The first one is <unk>, not counted in the actual UID.
            return cls(uids[1:])
        else:
            raise ValueError(f"不支持的加载方法: {method}")
