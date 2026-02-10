from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import TrainConfig, TARGET_BRANDS
from .data import CSVDataset
from .clip_utils import create_clip, encode_texts, encode_images
from .prompts import get_brand_prompt_map
from .metrics import binary_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _, preprocess_val = create_clip(args.model, args.pretrained, device)

    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    ds = CSVDataset(args.csv, transform=preprocess_val)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers)

    brand_prompts = get_brand_prompt_map()
    brand_text_emb = {}
    for brand, prompts in brand_prompts.items():
        t = encode_texts(model, tokenizer, prompts, device)
        brand_text_emb[brand] = F.normalize(t.mean(dim=0, keepdim=True), dim=-1)

    y_true, y_score = [], []
    for images, y_id, y_label in loader:
        images = images.to(device)
        img_feat = encode_images(model, images)

        target = (y_id != 0).int()   # background id is 0

        scores = []
        for brand in TARGET_BRANDS:
            cos = (img_feat @ brand_text_emb[brand].t()).squeeze(1)
            score01 = (cos + 1.0) / 2.0
            scores.append(score01)
        max_score = torch.stack(scores, dim=1).max(dim=1).values

        y_true.extend(target.cpu().numpy().tolist())
        y_score.extend(max_score.detach().cpu().numpy().tolist())

    m = binary_metrics(y_true, y_score, threshold=args.threshold)
    print(m)

if __name__ == "__main__":
    main()
