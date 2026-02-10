from __future__ import annotations
import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.predict import CLIPLogoPredictor

DEFAULT_CKPT = os.environ.get("LOGO_CKPT", "runs/exp1/best.pt")
MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("CLIP_PRETRAINED", "openai")
THRESHOLD = float(os.environ.get("CLIP_THRESHOLD", "0.61"))

app = FastAPI(title="CLIP Logo Detector")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

_predictor: CLIPLogoPredictor | None = None

@app.on_event("startup")
def _load():
    global _predictor
    if os.path.exists(DEFAULT_CKPT):
        _predictor = CLIPLogoPredictor(DEFAULT_CKPT, model_name=MODEL_NAME, pretrained=PRETRAINED, threshold=THRESHOLD)
    else:
        _predictor = None

@app.get("/", response_class=HTMLResponse)
def index():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _predictor is None:
        return {"error": f"Checkpoint not found at {DEFAULT_CKPT}. Train first or set LOGO_CKPT env var."}
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return _predictor.predict_pil(img)
