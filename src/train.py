from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .utils import set_seed
from .data import CSVDataset
from .clip_utils import create_clip
from .trainer import Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--freeze_encoders", action="store_true")
    ap.add_argument("--no_freeze_encoders", action="store_true")
    ap.add_argument("--supcon_weight", type=float, default=None)

    args = ap.parse_args()

    cfg = TrainConfig()
    cfg.model_name = args.model
    cfg.pretrained = args.pretrained

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr

    if args.no_freeze_encoders:
        cfg.freeze_encoders = False
    elif args.freeze_encoders:
        cfg.freeze_encoders = True

    if args.supcon_weight is not None:
        cfg.supcon_weight = args.supcon_weight

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, preprocess_train, preprocess_val = create_clip(cfg.model_name, cfg.pretrained, device)

    train_ds = CSVDataset(args.train_csv, transform=preprocess_train)
    val_ds = CSVDataset(args.val_csv, transform=preprocess_val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    trainer = Trainer(model, tokenizer, cfg, out_dir=args.out_dir)
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()
