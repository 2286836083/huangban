"""找改进最明显的图片：计算每张图 Exp-00(基线) vs Exp-07(集成+TTA) 的 AED 差距。"""
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from src.config import *
from src.model import HeatmapLocModel
from train_baseline_origin import OriginBaselineModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).parent


@torch.no_grad()
def predict_baseline(model, raw_img):
    img = cv2.resize(raw_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    coord = model(img)
    return coord[0].cpu().numpy()


@torch.no_grad()
def predict_ensemble(models, img_tensor):
    all_preds = []
    for m in models:
        coords, _ = m(img_tensor)
        # TTA flip
        cf, _ = m(img_tensor.flip(-1))
        cf[:, 0] = 1.0 - cf[:, 0]
        avg = (coords + cf) / 2.0
        all_preds.append(avg)
    return torch.stack(all_preds).mean(dim=0)[0].cpu().numpy()


def denormalize(norm_coord, params):
    px = (norm_coord[0] * 512 - params["pad_x"]) / (params["scale"] * params["inter_scale"])
    py = (norm_coord[1] * 512 - params["pad_y"]) / (params["scale"] * params["inter_scale"])
    return px, py


# Load models
print("Loading models...")
baseline_model = OriginBaselineModel().to(DEVICE)
ckpt = torch.load(SCRIPT_DIR / "output" / "checkpoints" / "baseline_origin" / "best_model.pth",
                  map_location=DEVICE, weights_only=True)
baseline_model.load_state_dict(ckpt["model_state_dict"])
baseline_model.eval()

ensemble_models = []
for k in range(1, 6):
    m = HeatmapLocModel("resnet34", pretrained=False).to(DEVICE)
    cp = SCRIPT_DIR / "checkpoints" / "5fold_heatmap" / "resnet34" / "resnet34" / f"fold_{k}" / "best_model.pth"
    ckpt = torch.load(cp, map_location=DEVICE, weights_only=True)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    ensemble_models.append(m)

# Evaluate all 100 images
gt_df = pd.read_excel(GT_FILE)
all_ids = sorted(gt_df["data"].tolist())

results = []
for pid in all_ids:
    row = gt_df[gt_df["data"] == pid].iloc[0]
    gt_x, gt_y = float(row["Fovea_X"]), float(row["Fovea_Y"])

    img_path = TRAIN_IMG_DIR / f"{pid:04d}.jpg"
    img_bytes = np.fromfile(img_path, dtype=np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Baseline prediction (native: coords are x/w, y/h)
    pred_baseline = predict_baseline(baseline_model, image)
    pred_baseline_orig = (pred_baseline[0] * w, pred_baseline[1] * h)
    ed_baseline = np.sqrt((pred_baseline_orig[0] - gt_x)**2 + (pred_baseline_orig[1] - gt_y)**2)
    aed_baseline = ed_baseline / np.sqrt(w * h)

    # Ensemble prediction
    # Prepare input with preprocessing
    orig_img = image
    inter_scale = 1.0
    if USE_PREPROCESSING and PREPROCESS_INTER_SIZE:
        max_dim = max(h, w)
        if max_dim > PREPROCESS_INTER_SIZE:
            inter_scale = PREPROCESS_INTER_SIZE / max_dim
            new_w = round(w * inter_scale)
            new_h = round(h * inter_scale)
            orig_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    from src.dataset import _fundus_preprocess, _aspect_preserve_resize_pad
    proc = _fundus_preprocess(orig_img)
    proc, (pad_x, pad_y, scale) = _aspect_preserve_resize_pad(proc, 512)

    # Normalize
    tensor = proc.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = (tensor - mean) / std
    tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    pred_ensemble_norm = predict_ensemble(ensemble_models, tensor)
    pred_ensemble_orig = denormalize(pred_ensemble_norm, {
        "pad_x": pad_x, "pad_y": pad_y, "scale": scale, "inter_scale": inter_scale})
    ed_ensemble = np.sqrt((pred_ensemble_orig[0] - gt_x)**2 + (pred_ensemble_orig[1] - gt_y)**2)
    aed_ensemble = ed_ensemble / np.sqrt(w * h)

    improvement = aed_baseline - aed_ensemble  # positive = baseline worse (good for showing improvement)
    results.append({
        "id": pid, "gt_x": gt_x, "gt_y": gt_y,
        "aed_baseline": aed_baseline, "aed_ensemble": aed_ensemble,
        "improvement": improvement, "ed_baseline_px": ed_baseline,
    })

# Sort by improvement (largest first)
df = pd.DataFrame(results).sort_values("improvement", ascending=False)
df["score_baseline"] = 1.0 / (df["aed_baseline"] + 0.1)
df["score_ensemble"] = 1.0 / (df["aed_ensemble"] + 0.1)

print("\nTop 10: Largest improvement (best for visualization)")
print(f"{'ID':<6} {'GT_x':<8} {'GT_y':<8} {'BL_AED':<8} {'ENS_AED':<8} {'Improv':<8} {'BL_Score':<8} {'ENS_Score':<8}")
for _, r in df.head(10).iterrows():
    print(f"{int(r['id']):04d} ({r['gt_x']:<6.0f},{r['gt_y']:<6.0f}) {r['aed_baseline']:<8.4f} {r['aed_ensemble']:<8.4f} {r['improvement']:<8.4f} {r['score_baseline']:<8.2f} {r['score_ensemble']:<8.2f}")

print("\nBottom 5: Smallest improvement")
for _, r in df.tail(5).iterrows():
    print(f"{int(r['id']):04d} ({r['gt_x']:<6.0f},{r['gt_y']:<6.0f}) {r['aed_baseline']:<8.4f} {r['aed_ensemble']:<8.4f} {r['improvement']:<8.4f}")

# Pick 4 diverse IDs (different fovea locations, large improvement)
top4 = df.head(8)
# Pick from different spatial regions
from sklearn.cluster import KMeans
coords = top4[["gt_x", "gt_y"]].values
km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(coords)
pick = []
for ci in range(4):
    candidates = top4.iloc[np.where(labels == ci)[0]]
    pick.append(int(candidates.iloc[0]["id"]))
pick.sort()
print(f"\nRecommended 4 sample IDs for visualization: {pick}")
