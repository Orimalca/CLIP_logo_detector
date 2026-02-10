from __future__ import annotations
import torch
import torch.nn.functional as F
import open_clip

def create_clip(model_name: str, pretrained: str, device: torch.device):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    return model, tokenizer, preprocess_train, preprocess_val

@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], device: torch.device) -> torch.Tensor:
    tok = tokenizer(texts).to(device)
    feat = model.encode_text(tok)
    return F.normalize(feat, dim=-1)

@torch.no_grad()
def encode_images(model, images: torch.Tensor) -> torch.Tensor:
    feat = model.encode_image(images)
    return F.normalize(feat, dim=-1)

def clip_contrastive_loss(image_feat: torch.Tensor, text_feat: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale * (image_feat @ text_feat.t())
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2

def supervised_contrastive_loss(z: torch.Tensor, y: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    B = z.size(0)
    sim = (z @ z.t()) / tau
    self_mask = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)

    y = y.view(-1, 1)
    pos_mask = (y == y.t()) & (~self_mask)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos_counts = pos_mask.sum(dim=1)
    # anchors with no positives contribute 0
    loss = torch.zeros(B, device=z.device)
    has_pos = pos_counts > 0
    loss[has_pos] = -(log_prob[has_pos] * pos_mask[has_pos]).sum(dim=1) / pos_counts[has_pos]
    return loss.mean()
