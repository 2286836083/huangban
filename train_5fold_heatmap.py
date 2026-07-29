"""
5-Fold CV Heatmap Model: ResNet34 (default)
============================================
- HeatmapLocModel: Backbone → Decoder → Heatmap → Soft-Argmax → (x, y)
- Full preprocessing: FOV → Illumination → CLAHE → Aspect-preserve
- Loss: MSE(heatmap) + MSE(coord)
- Aims for best single-fold Score > 9.0
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(max(1, os.cpu_count() // 2))
torch.backends.cudnn.benchmark = False

BACKBONE = "resnet34"
N_FOLDS = 5
EPOCHS = 150
BATCH_SIZE = 8
LR = 2e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
SEED = 42
ALL_IDS = list(range(1, 101))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    all_pred, all_gt, all_losses = [], [], []
    for images, heatmaps, coords in train_loader:
        images, heatmaps, coords = (images.to(device), heatmaps.to(device), coords.to(device))
        pred_coords, pred_hm = model(images)
        loss, ld = loss_fn(pred_hm, heatmaps, pred_coords, coords)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time.sleep(0.05)
        all_pred.append(pred_coords.detach().cpu())
        all_gt.append(coords.detach().cpu())
        all_losses.append(ld["loss"])
    pred_all = torch.cat(all_pred, dim=0).numpy()
    gt_all = torch.cat(all_gt, dim=0).numpy()
    from src.utils import compute_metrics
    metrics = compute_metrics(pred_all, gt_all)
    metrics["loss"] = float(np.mean(all_losses))
    return metrics


def validate(model, val_loader, loss_fn, device):
    model.eval()
    all_pred, all_gt, all_losses = [], [], []
    with torch.no_grad():
        for images, heatmaps, coords in val_loader:
            images, heatmaps, coords = (images.to(device), heatmaps.to(device), coords.to(device))
            pred_coords, pred_hm = model(images)
            loss, ld = loss_fn(pred_hm, heatmaps, pred_coords, coords)
            all_losses.append(ld["loss"])
            all_pred.append(pred_coords.cpu())
            all_gt.append(coords.cpu())
    pred_all = torch.cat(all_pred, dim=0).numpy()
    gt_all = torch.cat(all_gt, dim=0).numpy()
    from src.utils import compute_metrics
    metrics = compute_metrics(pred_all, gt_all)
    metrics["loss"] = float(np.mean(all_losses))
    return metrics


def train_one_fold(backbone_name, train_ids, val_ids, device, save_dir, gt_df, fold_k, img_dir):
    from src.dataset import HeatmapFundusDataset
    from src.model import HeatmapLocModel
    from src.loss import HeatmapLoss

    fold_dir = save_dir / f"fold_{fold_k}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  \u250c\u2500 Fold {fold_k} \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(f"  \u2502 Train: {len(train_ids)} samples, Val: {len(val_ids)} samples")

    train_dataset = HeatmapFundusDataset(
        img_dir, gt_df, train_ids, mode="train", augment=True)
    val_dataset = HeatmapFundusDataset(
        img_dir, gt_df, val_ids, mode="val", augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False, drop_last=False)

    model = HeatmapLocModel(backbone_name=backbone_name, pretrained=True).to(device)
    loss_fn = HeatmapLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    best_aed = float("inf")
    best_epoch = -1
    no_improve = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        epoch_time = time.time() - epoch_t0

        aed = val_metrics["ED_mean"]
        score = 1.0 / (aed + 0.1)
        history.append({
            "epoch": epoch, "val_AED": aed, "val_median": val_metrics["ED_median"],
            "val_max": val_metrics["ED_max"], "val_score": score,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"  \u2502 Epoch {epoch:03d} | Val AED: {aed:.4f} | Score: {score:.2f} | {epoch_time:.1f}s")

        if aed < best_aed:
            best_aed = aed
            best_epoch = epoch
            no_improve = 0
            from src.utils import save_checkpoint
            save_checkpoint(model, optimizer, epoch, val_metrics,
                            is_best=True, save_dir=fold_dir)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  \u2502 Early stop @ epoch {epoch} (patience={PATIENCE})")
                break

    total_time = time.time() - t_start
    print(f"  \u2502 Best AED: {best_aed:.4f} | Score: {1/(best_aed+0.1):.2f} @ epoch {best_epoch} ({total_time:.0f}s)")
    print(f"  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    return {
        "backbone": backbone_name, "fold": fold_k,
        "best_aed": best_aed, "best_epoch": best_epoch,
        "score": 1.0 / (best_aed + 0.1), "total_time": total_time,
    }, history


def main():
    import src.config
    src.config.USE_PREPROCESSING = True
    src.config.USE_FOV_MASK = True
    src.config.USE_ILLUM_CORRECTION = True
    src.config.USE_CLAHE = True
    src.config.USE_ASPECT_PRESERVE = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Heatmap 5-fold CV | Backbone: {BACKBONE}")

    SCRIPT_DIR = Path(__file__).parent
    CP_DIR = SCRIPT_DIR / "checkpoints" / "5fold_heatmap"
    FIG_DIR = SCRIPT_DIR / "output"
    CP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, val_idx in kf.split(ALL_IDS):
        folds.append(([ALL_IDS[i] for i in train_idx], [ALL_IDS[i] for i in val_idx]))

    gt_df = pd.read_excel(src.config.GT_FILE)
    all_results = []
    t_total_start = time.time()

    from src.dataset import HeatmapFundusDataset
    model_sizes = {}
    for k, (train_ids, val_ids) in enumerate(folds, 1):
        HeatmapFundusDataset.clear_cache()

        fold_dir = CP_DIR / BACKBONE / f"fold_{k}"
        if (fold_dir / "best_model.pth").exists() and (fold_dir / "history.csv").exists():
            hist_df = pd.read_csv(fold_dir / "history.csv")
            best_row = hist_df.loc[hist_df["val_AED"].idxmin()]
            print(f"  \u23ed\ufe0f  Fold {k} already complete (AED={best_row['val_AED']:.4f}), skipping")
            all_results.append({
                "backbone": BACKBONE, "fold": k,
                "best_aed": best_row["val_AED"], "best_epoch": int(best_row["epoch"]),
                "score": 1.0 / (best_row["val_AED"] + 0.1), "total_time": 0,
            })
            continue

        if not model_sizes:
            from src.model import HeatmapLocModel
            m = HeatmapLocModel(backbone_name=BACKBONE, pretrained=False)
            model_sizes["params"] = count_parameters(m)
            print(f"  Parameters: {model_sizes['params']:,}")
            del m

        res, _ = train_one_fold(BACKBONE, train_ids, val_ids,
                                device, fold_dir / ".." / BACKBONE, gt_df, k,
                                src.config.TRAIN_IMG_DIR)
        all_results.append(res)

    t_total = time.time() - t_total_start
    print(f"\n\n{'=' * 70}")
    print(f"Total training time: {t_total:.0f}s")

    df_results = pd.DataFrame(all_results)

    print(f"\n{'=' * 80}")
    print(f"{'Backbone':<12} {'Fold':>5}  {'AED':>8}  {'Score':>7}  {'Best Ep':>7}  {'Time':>7}")
    print(f"{'-'*80}")
    for _, row in df_results.iterrows():
        print(f"{row['backbone']:<12} {row['fold']:>5d}  "
              f"{row['best_aed']:>8.4f}  {row['score']:>7.2f}  "
              f"{row['best_epoch']:>7d}  {row['total_time']:>7.0f}s")

    print(f"\n{'=' * 70}")
    agg = df_results.groupby("backbone")["best_aed"].agg(["mean", "std", "min"])
    score_agg = df_results.groupby("backbone")["score"].agg(["mean", "std"])
    print(f"{'Backbone':<12} {'AED_mean':>10} {'AED_std':>9} {'AED_min':>9} "
          f"{'Score_m':>9} {'Score_std':>9}")
    print(f"{'-'*60}")
    for bb in [BACKBONE]:
        print(f"{bb:<12}  {agg.loc[bb,'mean']:>8.4f}   {agg.loc[bb,'std']:>7.4f}  "
              f"{agg.loc[bb,'min']:>8.4f}   {score_agg.loc[bb,'mean']:>7.2f}   "
              f"{score_agg.loc[bb,'std']:>7.2f}")

    df_results.to_csv(CP_DIR / "5fold_summary.csv", index=False)
    print(f"\n  Results saved to: {CP_DIR}")


if __name__ == "__main__":
    main()
