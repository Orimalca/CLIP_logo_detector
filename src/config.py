from __future__ import annotations
from dataclasses import dataclass

TARGET_BRANDS = ["cocacola", "disney", "mcdonalds", "starbucks"]
ALL_LABELS = ["background"] + TARGET_BRANDS

@dataclass
class TrainConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"

    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 5
    seed: int = 42

    freeze_encoders: bool = True
    supcon_weight: float = 0.0
    supcon_tau: float = 0.07

    threshold: float = 0.61
