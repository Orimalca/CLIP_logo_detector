from __future__ import annotations
import torch
import torch.nn.functional as F
from PIL import Image

from .clip_utils import create_clip, encode_texts
from .prompts import get_brand_prompt_map
from .config import TARGET_BRANDS

class CLIPLogoPredictor:
    def __init__(self, ckpt_path: str, model_name: str = "ViT-B-32", pretrained: str = "openai", threshold: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer, _, self.preprocess = create_clip(model_name, pretrained, self.device)

        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()

        self.threshold = threshold

        brand_prompts = get_brand_prompt_map()
        self.brand_text = {}
        for brand, prompts in brand_prompts.items():
            t = encode_texts(self.model, self.tokenizer, prompts, self.device)
            self.brand_text[brand] = F.normalize(t.mean(dim=0, keepdim=True), dim=-1)

    @torch.no_grad()
    def predict_pil(self, img: Image.Image) -> dict:
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        img_feat = F.normalize(self.model.encode_image(x), dim=-1)

        scores = {}
        best_brand, best_score = None, -1.0
        for brand in TARGET_BRANDS:
            cos = (img_feat @ self.brand_text[brand].t()).item()
            score01 = (cos + 1.0) / 2.0
            scores[brand] = float(score01)
            if score01 > best_score:
                best_score = score01
                best_brand = brand

        pred = 1 if best_score >= self.threshold else 0
        return {
            "prediction": pred,
            "best_brand": best_brand,
            "best_score": float(best_score),
            "threshold": float(self.threshold),
            "scores": scores,
        }
