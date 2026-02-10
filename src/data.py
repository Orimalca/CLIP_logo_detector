from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

from .config import ALL_LABELS

LABEL_TO_ID = {name: i for i, name in enumerate(ALL_LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

@dataclass
class Sample:
    path: str
    label: str

class CSVDataset(Dataset):
    def __init__(self, csv_path: str, transform: Optional[Callable] = None):
        self.transform = transform
        self.samples: list[Sample] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if r.fieldnames is None or "path" not in r.fieldnames or "label" not in r.fieldnames:
                raise ValueError("CSV must have columns: path,label")
            for row in r:
                lab = row["label"].strip().lower()
                if lab not in LABEL_TO_ID:
                    raise ValueError(f"Unknown label in CSV: {lab}. Allowed: {list(LABEL_TO_ID.keys())}")
                self.samples.append(Sample(path=row["path"], label=lab))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img) if self.transform is not None else img
        y = LABEL_TO_ID[s.label]
        return x, y, s.label
