"""
Architecture Ablation: Simple Baseline (ResNet18)
=================================================
- Backbone → GAP → FC(512→2) → Sigmoid
- No SE, no SpatialAttn, no Mixup
- Same 70/30 split (0001-0070 / 0071-0100) as Exp-01
- Isolates: how much score comes from architecture vs. from raw data?
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(max(1, os.cpu_count() // 2))
torch.backends.cudnn.benchmark = False

from src.utils import compute_metrics, save_checkpoint

BACKBONE = "resnet18"
EPOCHS = 150
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
SEED = 42
TRAIN_IDS = list(range(1, 71))
VAL_IDS = list(range(71, 101))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    all_pred, all_gt, all_losses = [], [], []
    for images, coords in train_loader:
        images, coords = images.to(device), coords.to(device)
        pred_coords = model(images)
        loss, ld = loss_fn(pred_coords, coords)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time.sleep(0.05)
        all_pred.append(pred_coords.detach().cpu())
        all_gt.append(coords.detach().cpu())
        all_losses.append(ld["loss"])
    pred_all = torch.cat(all_pred, dim=0).numpy()
    gt_all = torch.cat(all_gt, dim=0).numpy()
    metrics = compute_metrics(pred_all, gt_all)
    metrics["loss"] = float(np.mean(all_losses))
    return metrics


def validate(model, val_loader, loss_fn, device):
    model.eval()
    all_pred, all_gt, all_losses = [], [], []
    with torch.no_grad():
        for images, coords in val_loader:
            images, coords = images.to(device), coords.to(device)
            pred_coords = model(images)
            loss, ld = loss_fn(pred_coords, coords)
            all_losses.append(ld["loss"])
            all_pred.append(pred_coords.cpu())
            all_gt.append(coords.cpu())
    pred_all = torch.cat(all_pred, dim=0).numpy()
    gt_all = torch.cat(all_gt, dim=0).numpy()
    metrics = compute_metrics(pred_all, gt_all)
    metrics["loss"] = float(np.mean(all_losses))
    return metrics


def main():
    import src.config
    # Raw data: no preprocessing
    src.config.USE_PREPROCESSING = False
    src.config.USE_FOV_MASK = False
    src.config.USE_ILLUM_CORRECTION = False
    src.config.USE_CLAHE = False
    # No Mixup
    src.config.MIXUP_ALPHA = 0.0
    src.config.MIXUP_PROB = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Simple baseline: {BACKBONE} (no SE, no SpatialAttn, no Mixup)")
    print(f"Raw data: {len(TRAIN_IDS)} train / {len(VAL_IDS)} val (70/30 split)")

    SCRIPT_DIR = Path(__file__).parent
    CP_DIR = SCRIPT_DIR / "checkpoints" / "simple_baseline"
    FIG_DIR = SCRIPT_DIR / "output"
    CP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    from src.dataset import FundusDataset
    from src.model import SimpleFoveaModel
    from src.loss import FoveaLoss

    FundusDataset.clear_cache()

    gt_df = pd.read_excel(src.config.GT_FILE)

    train_dataset = FundusDataset(src.config.TRAIN_IMG_DIR, gt_df, TRAIN_IDS,
                                  mode="train", augment=True)
    val_dataset = FundusDataset(src.config.TRAIN_IMG_DIR, gt_df, VAL_IDS,
                                mode="val", augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False, drop_last=False)

    model = SimpleFoveaModel(backbone_name=BACKBONE, pretrained=True).to(device)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    loss_fn = FoveaLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    best_aed = float("inf")
    best_epoch = -1
    best_score = 0.0
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
            "epoch": epoch,
            "val_AED": aed,
            "val_median": val_metrics["ED_median"],
            "val_max": val_metrics["ED_max"],
            "val_std": val_metrics["ED_std"],
            "val_score": score,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | Val AED: {aed:.4f} | Score: {score:.2f} | {epoch_time:.1f}s")

        if aed < best_aed:
            best_aed = aed
            best_epoch = epoch
            best_score = score
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_metrics,
                            is_best=True, save_dir=CP_DIR)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop @ epoch {epoch} (patience={PATIENCE})")
                break

    total_time = time.time() - t_start
    print(f"\n{'=' * 50}")
    print(f"  Best AED: {best_aed:.4f}  |  Best Score: {best_score:.2f}")
    print(f"  Best epoch: {best_epoch}  |  Time: {total_time:.0f}s")
    print(f"{'=' * 50}")

    pd.DataFrame(history).to_csv(CP_DIR / "history.csv", index=False)
    print(f"  History saved to: {CP_DIR / 'history.csv'}")

    # Convergence plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist_df = pd.DataFrame(history)
    plt.figure(figsize=(10, 5))
    plt.plot(hist_df["epoch"], hist_df["val_AED"], label="Simple Baseline (ResNet18)",
             color="#E74C3C", linewidth=2)
    plt.axhline(0.01433, color="#2E86AB", linestyle="--", alpha=0.7,
                label=f"Exp-01 ResNet18 (best AED=0.01433)")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Validation AED", fontsize=14)
    plt.title("Simple Baseline vs Enhanced Architecture (Exp-01)", fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "simple_baseline_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to: {FIG_DIR / 'simple_baseline_convergence.png'}")

    print(f"\n  Comparison with Exp-01 (ResNet18):")
    print(f"    Enhanced: AED=0.01433, Score=8.75")
    print(f"    Simple:   AED={best_aed:.4f}, Score={best_score:.2f}")
    print(f"    Delta:    AED={best_aed - 0.01433:+.4f}")


if __name__ == "__main__":
    main()
