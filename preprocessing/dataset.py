import numpy as np
import json
import os
import pandas as pd
import pickle
import random
import torch
from typing import Iterable, List, Optional
from enum import Enum
import re
from datasets import Dataset


class BaseDataset:
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        seed: Optional[int] = None,
    ) -> None:
        self.train_data: BaseDataset = None
        self.val_data: BaseDataset = None
        self.test_data: BaseDataset = None
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.seed = seed
        self._set_seeds()
        self._set_train_data()
        self._set_val_data()
        self._set_test_data()
        self.name = None

    def _set_seeds(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def get_name(self):
        return self.name

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data

    def _set_train_data(self) -> None:
        """Set the self.train_data field to the dataset."""
        train_df = pd.read_csv(self.train_path)
        self.train_data = Dataset.from_pandas(train_df, split="train", preserve_index=False)

    def _set_val_data(self) -> None:
        """Set the self.val_data field to the dataset."""
        val_df = pd.read_csv(self.val_path)
        self.val_data = Dataset.from_pandas(val_df, split="val", preserve_index=False)

    def _set_test_data(self) -> None:
        """Set the self.test_data field to the dataset."""
        test_df = pd.read_csv(self.test_path)
        self.test_data = Dataset.from_pandas(test_df, split="test", preserve_index=False)
