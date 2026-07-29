"""
Backbone Comparison Experiment: ResNet18 vs ResNet34 vs ResNet50
===============================================================
- Raw data mode: NO preprocessing (no FOV, illumination, CLAHE)
- Fixed 70/30 split (train: 0001-0070, val: 0071-0100)
- Identical hyperparameters for all backbones
- Output: comparison table + convergence curves + per-backbone checkpoints
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Thermal protection
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(max(1, os.cpu_count() // 2))
torch.backends.cudnn.benchmark = False

from src.utils import compute_metrics, save_checkpoint

# ================================================================
#  Hyperparameters (identical across backbones)
# ================================================================
BACKBONES = ["resnet18", "resnet34", "resnet50"]
EPOCHS = 150
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.5
SEED = 42

# Fixed split (no randomization — deterministic comparison)
TRAIN_IDS = list(range(1, 71))
VAL_IDS   = list(range(71, 101))

# Colors for plotting
COLORS = {"resnet18": "#2E86AB", "resnet34": "#A23B72", "resnet50": "#F18F01"}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    all_pred, all_gt, all_losses = [], [], []
    from src.dataset import mixup_batch
    for batch_idx, (images, coords) in enumerate(train_loader):
        images, coords = images.to(device), coords.to(device)
        if MIXUP_ALPHA > 0 and np.random.random() < MIXUP_PROB:
            images, coords, _ = mixup_batch(images, coords)
        pred_coords = model(images)
        loss, ld = loss_fn(pred_coords, coords)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time.sleep(0.05)  # Thermal: micro-break between batches
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


def train_backbone(backbone_name, device, cp_dir, gt_df):
    """Train a single backbone and return (results_dict, history)."""
    import src.config
    from src.dataset import FundusDataset
    from src.model import FoveaLocalizationModel
    from src.loss import FoveaLoss
    print(f"\n{'=' * 60}")
    print(f"Backbone: {backbone_name}")
    print(f"{'=' * 60}")

    # ---- Data (num_workers=0 for Windows compatibility) ----
    train_dataset = FundusDataset(
        src.config.TRAIN_IMG_DIR, gt_df, TRAIN_IDS,
        mode="train", augment=True,
    )
    val_dataset = FundusDataset(
        src.config.TRAIN_IMG_DIR, gt_df, VAL_IDS,
        mode="val", augment=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False,
    )

    # ---- Model ----
    model = FoveaLocalizationModel(backbone_name=backbone_name, pretrained=True).to(device)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    loss_fn = FoveaLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---- Training loop ----
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
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_AED": train_metrics["ED_mean"],
            "val_loss": val_metrics["loss"],
            "val_AED": aed,
            "val_median": val_metrics["ED_median"],
            "val_max": val_metrics["ED_max"],
            "val_std": val_metrics["ED_std"],
        })

        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:03d} | Train AED: {train_metrics['ED_mean']:.4f} | "
              f"Val AED: {aed:.4f} | LR: {lr_now:.2e} | {epoch_time:.1f}s")

        if aed < best_aed:
            best_aed = aed
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_metrics,
                            is_best=True, save_dir=cp_dir / backbone_name)
            print(f"    >> New best! AED = {best_aed:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    Early stop at epoch {epoch} (patience={PATIENCE})")
                break

    total_time = time.time() - t_start
    print(f"  Done. Best AED = {best_aed:.4f} @ epoch {best_epoch} ({total_time:.0f}s)")

    # Save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(cp_dir / backbone_name / "history.csv", index=False)

    return {
        "backbone": backbone_name,
        "params": n_params,
        "best_aed": best_aed,
        "best_epoch": best_epoch,
        "best_score": 1.0 / (best_aed + 0.1),
        "best_median": min(h["val_median"] for h in history),
        "total_time": total_time,
    }, history


def plot_convergence(all_history, save_path):
    """Plot validation AED curves for all backbones on one graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for backbone_name, hist in all_history.items():
        df = pd.DataFrame(hist)
        plt.plot(df["epoch"], df["val_AED"],
                 label=backbone_name, color=COLORS[backbone_name], linewidth=2)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Validation AED", fontsize=14)
    plt.title("Backbone Comparison — Validation AED Convergence", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Convergence plot saved to: {save_path}")


def main():
    # ================================================================
    # IMPORTANT: Override config BEFORE importing dataset/model
    # ================================================================
    import src.config
    src.config.USE_PREPROCESSING = False
    src.config.USE_FOV_MASK = False
    src.config.USE_ILLUM_CORRECTION = False
    src.config.USE_CLAHE = False
    src.config.USE_ASPECT_PRESERVE = True
    src.config.USE_AUGMENTATION = True

    # Now import project modules (sees overridden config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Raw data mode: preprocessing disabled (FOV/Illum/CLAHE all OFF)")

    # Output directories
    SCRIPT_DIR = Path(__file__).parent
    CP_DIR = SCRIPT_DIR / "checkpoints" / "comparison"
    CP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR = SCRIPT_DIR / "output"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Set seed ----
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Load GT once ----
    gt_df = pd.read_excel(src.config.GT_FILE)

    # ---- Train all backbones ----
    results = []
    all_history = {}

    for backbone_name in BACKBONES:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        res, history = train_backbone(backbone_name, device, CP_DIR, gt_df)
        results.append(res)
        all_history[backbone_name] = history

    # ---- Summary table ----
    print(f"\n{'=' * 70}")
    print("Backbone Comparison — Summary")
    print(f"{'=' * 70}")

    df_results = pd.DataFrame(results)
    df_results = df_results[[
        "backbone", "params", "best_aed", "best_score",
        "best_median", "best_epoch", "total_time",
    ]]
    print("\n")
    print(df_results.to_string(index=False))

    csv_path = CP_DIR / "comparison_summary.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Comparison table saved to: {csv_path}")

    # ---- Plot ----
    plot_path = FIG_DIR / "backbone_comparison_convergence.png"
    plot_convergence(all_history, plot_path)

    # ---- Analysis ----
    print(f"\n{'─' * 70}")
    best_row = df_results.loc[df_results["best_score"].idxmax()]
    print(f"  Best backbone by Score: {best_row['backbone']} "
          f"(AED={best_row['best_aed']:.4f}, Score={best_row['best_score']:.2f})")
    print(f"\n  Done! All checkpoints saved to: {CP_DIR}")


if __name__ == "__main__":
    main()
