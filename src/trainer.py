from __future__ import annotations
import os
import json
from dataclasses import asdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import ensure_dir, get_device
from .config import TrainConfig, ALL_LABELS, TARGET_BRANDS
from .prompts import sample_prompt, get_brand_prompt_map
from .clip_utils import clip_contrastive_loss, supervised_contrastive_loss, encode_texts, encode_images
from .metrics import binary_metrics

class Trainer:
    def __init__(self, model, tokenizer, cfg: TrainConfig, out_dir: str):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.out_dir = out_dir
        ensure_dir(out_dir)

        self.device = get_device()
        self.model.to(self.device)

        if cfg.freeze_encoders:
            for name, p in self.model.named_parameters():
                if "logit_scale" in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("No trainable parameters. Use --no_freeze_encoders or keep logit_scale trainable.")
        self.optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.best_f1 = -1.0

        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

    def _make_text_batch(self, labels: list[str], rng) -> torch.Tensor:
        texts = [sample_prompt(lab, rng) for lab in labels]
        return self.tokenizer(texts).to(self.device)

    @torch.no_grad()
    def eval_binary(self, loader: DataLoader, threshold: float) -> dict:
        self.model.eval()

        brand_prompts = get_brand_prompt_map()
        brand_text_emb = {}
        for brand, prompts in brand_prompts.items():
            t = encode_texts(self.model, self.tokenizer, prompts, self.device)
            t = F.normalize(t.mean(dim=0, keepdim=True), dim=-1)
            brand_text_emb[brand] = t

        target_ids = torch.tensor([ALL_LABELS.index(b) for b in TARGET_BRANDS], device=self.device)

        y_true, y_score = [], []
        for images, y_id, y_label in loader:
            images = images.to(self.device)
            y_id = y_id.to(self.device)

            img_feat = encode_images(self.model, images)

            target = torch.isin(y_id, target_ids).int()

            scores = []
            for brand in TARGET_BRANDS:
                cos = (img_feat @ brand_text_emb[brand].t()).squeeze(1)
                score01 = (cos + 1.0) / 2.0
                scores.append(score01)
            max_score = torch.stack(scores, dim=1).max(dim=1).values

            y_true.extend(target.detach().cpu().numpy().tolist())
            y_score.extend(max_score.detach().cpu().numpy().tolist())

        return binary_metrics(y_true, y_score, threshold=threshold)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        import random
        rng = random.Random(self.cfg.seed)

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{self.cfg.epochs}")
            losses = []

            for images, y_id, y_label in pbar:
                images = images.to(self.device)
                y_id = y_id.to(self.device)

                tok = self._make_text_batch(list(y_label), rng)

                image_feat = F.normalize(self.model.encode_image(images), dim=-1)
                text_feat = F.normalize(self.model.encode_text(tok), dim=-1)

                logit_scale = self.model.logit_scale.exp()
                loss = clip_contrastive_loss(image_feat, text_feat, logit_scale)

                if self.cfg.supcon_weight > 0:
                    z_for_sup = image_feat if not self.cfg.freeze_encoders else image_feat.detach()
                    loss_sup = supervised_contrastive_loss(z_for_sup, y_id, tau=self.cfg.supcon_tau)
                    loss = loss + self.cfg.supcon_weight * loss_sup

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                self.optim.step()

                losses.append(float(loss.item()))
                pbar.set_postfix(loss=sum(losses)/max(len(losses), 1))

            val_m = self.eval_binary(val_loader, threshold=self.cfg.threshold)

            torch.save({"model": self.model.state_dict()}, os.path.join(self.out_dir, "last.pt"))

            if val_m["f1"] > self.best_f1:
                self.best_f1 = val_m["f1"]
                torch.save({"model": self.model.state_dict(), "val_metrics": val_m}, os.path.join(self.out_dir, "best.pt"))

            with open(os.path.join(self.out_dir, "metrics.txt"), "a", encoding="utf-8") as f:
                f.write(f"epoch={epoch} train_loss={sum(losses)/len(losses):.4f} val={val_m}\n")

        return {"best_f1": self.best_f1}
