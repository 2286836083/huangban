"""
5-Fold Ensemble + TTA Evaluation
=================================
Loads all 5 ResNet34 heatmap checkpoints, runs out-of-fold ensemble
with test-time augmentation (horizontal flip), evaluates overall metrics.

Usage:
    cd d:\OneDrive\桌面\最终\新工作区
    python ensemble_eval.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(max(1, os.cpu_count() // 2))

BACKBONE = "resnet34"
N_FOLDS = 5
SEED = 42
ALL_IDS = list(range(1, 101))
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(ckpt_base):
    from src.model import HeatmapLocModel

    models = []
    for k in range(1, N_FOLDS + 1):
        cp_path = ckpt_base / f"fold_{k}" / "best_model.pth"
        if not cp_path.exists():
            print(f"  [WARN] Checkpoint not found: {cp_path}")
            models.append(None)
            continue

        model = HeatmapLocModel(backbone_name=BACKBONE, pretrained=False).to(DEVICE)
        ckpt = torch.load(cp_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        print(f"  Loaded fold {k} (epoch {ckpt.get('epoch', '?')}, AED={ckpt.get('metrics', {}).get('ED_mean', '?'):.4f})")

    return models


@torch.no_grad()
def predict_tta(model, images):
    """Run inference with TTA (horizontal flip + average).

    Args:
        model: nn.Module
        images: [B, 3, H, W] tensor on device

    Returns:
        coords: [B, 2] tensor on device (normalized [0,1])
    """
    # Original
    coords_orig, _ = model(images)

    # Horizontal flip
    images_flip = images.flip(-1)
    coords_flip, _ = model(images_flip)
    coords_flip[:, 0] = 1.0 - coords_flip[:, 0]  # flip x back

    coords_avg = (coords_orig + coords_flip) / 2.0
    return coords_avg


def get_fold_assignment(n_ids=100, n_folds=5, seed=42):
    """Reconstruct fold assignment matching train_5fold_heatmap.py."""
    all_ids = list(range(1, n_ids + 1))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_id = {}
    for fold_k, (_, val_idx) in enumerate(kf.split(all_ids)):
        for i in val_idx:
            fold_of_id[all_ids[i]] = fold_k + 1
    return fold_of_id


def main():
    import src.config
    from src.dataset import HeatmapFundusDataset
    from src.utils import compute_metrics

    SCRIPT_DIR = Path(__file__).parent
    CKPT_DIR = SCRIPT_DIR / "checkpoints" / "5fold_heatmap" / BACKBONE / BACKBONE
    img_dir = src.config.TRAIN_IMG_DIR
    gt_df = pd.read_excel(src.config.GT_FILE)
    fold_of_id = get_fold_assignment()

    print(f"{'=' * 70}")
    print(f"  5-Fold Ensemble + TTA Evaluation")
    print(f"  Backbone: {BACKBONE}")
    print(f"  Device: {DEVICE}")
    print(f"{'=' * 70}")

    # Load models
    print(f"\n[1/3] Loading {N_FOLDS} models...")
    models = load_models(CKPT_DIR)
    available = [m for m in models if m is not None]
    print(f"  {len(available)}/{N_FOLDS} models loaded")

    if len(available) < 2:
        print("  Need at least 2 models for ensemble, abort.")
        return

    # Build dataset for all 100 images
    print(f"\n[2/3] Evaluating {len(ALL_IDS)} samples with ensemble + TTA...")
    dataset = HeatmapFundusDataset(img_dir, gt_df, ALL_IDS, mode="val", augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    all_preds = []
    all_gts = []
    per_fold_results = {k: {"preds": [], "gts": []} for k in range(1, N_FOLDS + 1)}
    t0 = time.time()

    for batch_idx, (images, _, coords) in enumerate(loader):
        idx = ALL_IDS[batch_idx]
        fold_k = fold_of_id[idx]
        images = images.to(DEVICE)
        coords_np = coords.numpy()[0]

        # Ensemble: use models that did NOT train on this fold
        ens_preds = []
        for k in range(1, N_FOLDS + 1):
            if k == fold_k:
                continue  # skip the model trained on this sample
            model = models[k - 1]
            if model is None:
                continue
            pred = predict_tta(model, images)
            ens_preds.append(pred.cpu().numpy()[0])

        if len(ens_preds) == 0:
            continue

        avg_pred = np.mean(ens_preds, axis=0)
        all_preds.append(avg_pred)
        all_gts.append(coords_np)
        per_fold_results[fold_k]["preds"].append(avg_pred)
        per_fold_results[fold_k]["gts"].append(coords_np)

    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    total_time = time.time() - t0

    # [3/3] Compute metrics
    print(f"\n[3/3] Computing metrics...")
    print(f"\n{'=' * 70}")
    print(f"  Ensemble + TTA Results (out-of-fold, n={len(all_preds)})")
    print(f"{'=' * 70}")

    # Overall metrics
    overall = compute_metrics(all_preds, all_gts)
    aed = overall["ED_mean"]
    score = 1.0 / (aed + 0.1)
    print(f"  Overall AED:    {aed:.4f}")
    print(f"  Overall Score:  {score:.2f}")
    print(f"  Median:         {overall['ED_median']:.4f}")
    print(f"  Max:            {overall['ED_max']:.4f}")
    print(f"  Min:            {overall['ED_min']:.4f}")
    print(f"  Std:            {overall['ED_std']:.4f}")
    print(f"  Time:           {total_time:.1f}s")

    # Per-fold metrics
    print(f"\n{'─' * 70}")
    print(f"  Per-fold breakdown")
    print(f"{'─' * 70}")
    fold_summary = []
    for k in range(1, N_FOLDS + 1):
        data = per_fold_results[k]
        if len(data["preds"]) == 0:
            continue
        fold_metrics = compute_metrics(np.array(data["preds"]), np.array(data["gts"]))
        faed = fold_metrics["ED_mean"]
        fscore = 1.0 / (faed + 0.1)
        fold_summary.append({"fold": k, "AED": faed, "Score": fscore, "n": len(data["preds"])})
        print(f"  Fold {k}: AED={faed:.4f}  Score={fscore:.2f}  (n={len(data['preds'])})")

    # Compare with single-model best
    print(f"\n{'─' * 70}")
    print(f"  Comparison")
    print(f"{'─' * 70}")
    print(f"  Single-model best fold:    AED=0.0055  Score=9.48 (ResNet34 heatmap Fold 2)")
    print(f"  Single-model mean (5-fold): AED=0.0095  Score=9.14")
    print(f"  Ensemble + TTA (out-of-fold): AED={aed:.4f}  Score={score:.2f}")
    improvement = (0.0095 - aed) / 0.0095 * 100
    print(f"  Improvement vs single-model mean: AED ↓ {improvement:.1f}%")

    # Save results
    print(f"\n{'─' * 70}")
    results = {
        "method": "5-fold_ensemble_TTA",
        "backbone": BACKBONE,
        "AED": aed,
        "Score": score,
        "ED_median": overall["ED_median"],
        "ED_std": overall["ED_std"],
        "n_samples": len(all_preds),
        "time_s": total_time,
    }
    results_df = pd.DataFrame([results])
    out_dir = SCRIPT_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "ensemble_results.csv", index=False)
    print(f"  Results saved to: {out_dir / 'ensemble_results.csv'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
