# CLIP Logo Detector (Binary: target-brand logo vs not)

### Task
- Input: an image
- Output 1 if it contains a logo of ANY of these brands: Coca-Cola, Disney, Starbucks, McDonald's. Otherwise output 0.

### Quick start

Note: use Python 3.10 or higher.

1) Environment Installation
```bash
virtualenv -p python3.10 .venv # Or alternatively use `python -m venv .venv`
source .venv/bin/activate
pip install -r requirements.txt
```

2) Download dataset
```bash
gdown --id 1DOg4ENNFu9JrXkGSFS4IQFDiBlK41QtU  # Download zip file from Google Drive
unzip -o ./dataset.zip -d dataset  # Extract contents of the zip file
```

3) Build splits (train/val/test CSVs)
```bash
python scripts/build_splits_from_folders.py --dataset_root dataset --out_dir data --tiny
```

This writes:
  data/train.csv, data/val.csv, data/test.csv
CSV columns:
  path,label
where label is one of: background,cocacola,disney,mcdonalds,starbucks

4) Train (CLIP contrastive)
```bash
python -m src.train --train_csv data/train.csv --val_csv data/val.csv --out_dir runs/exp1 --epochs 2 --batch_size 32 --model ViT-B-32 --pretrained openai --freeze_encoders
```

Notes:
- Training uses the standard CLIP image-text contrastive loss.
- Optional extra: supervised image-image contrastive loss (SupCon) with --supcon_weight > 0

5) Evaluate (binary metrics)
```bash
python -m src.evaluate --ckpt runs/exp1/best.pt --csv data/test.csv --threshold 0.61
```

Threshold meaning
We use cosine similarity mapped to `[0,1]` via:
  `score = (cos_sim + 1) / 2`
So `threshold=0.61` corresponds to `cos_sim >= 0.22`

Inference rule (what the web app uses)
- Compute CLIP image embedding
- Compute CLIP text embeddings for prompts like "disney logo", "mcdonalds logo", ...
- For each brand, average prompt embeddings, then compute `score = (cos+1)/2`
- If `max_brand_score >= threshold` -> `prediction=1` else `0`
- Also returns the best brand + scores

6) Run the web app
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:
  http://localhost:8000

REST:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@/path/to/image.jpg"
```

Docker (optional)
```bash
docker build -t clip-logo-detector .
docker run -p 8000:8000 -e LOGO_CKPT=runs/exp1/best.pt clip-logo-detector
```