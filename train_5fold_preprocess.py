"""
5-Fold CV with Preprocessing: ResNet18 vs ResNet34
===================================================
- Full preprocessing: FOV → Illumination → CLAHE → Aspect-preserve
- Same 5-fold splits (seed=42) as Exp-02 → direct A/B comparison
- Only ResNet18 + ResNet34 (ResNet50 dropped per Exp-02 findings)
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

from src.utils import compute_metrics, save_checkpoint

BACKBONES = ["resnet18", "resnet34"]
N_FOLDS = 5
EPOCHS = 150
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.5
SEED = 42
ALL_IDS = list(range(1, 101))
COLORS = {"resnet18": "#2E86AB", "resnet34": "#A23B72"}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    all_pred, all_gt, all_losses = [], [], []
    from src.dataset import mixup_batch
    for images, coords in train_loader:
        images, coords = images.to(device), coords.to(device)
        if MIXUP_ALPHA > 0 and np.random.random() < MIXUP_PROB:
            images, coords, _ = mixup_batch(images, coords)
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


def train_one_fold(backbone_name, train_ids, val_ids, device, save_dir, gt_df, fold_k):
    import src.config
    from src.dataset import FundusDataset
    from src.model import FoveaLocalizationModel
    from src.loss import FoveaLoss

    fold_dir = save_dir / f"fold_{fold_k}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  ┌─ Fold {fold_k} ─────────────────────────────────────────")
    print(f"  │ Train: {len(train_ids)} samples, Val: {len(val_ids)} samples")

    train_dataset = FundusDataset(
        src.config.TRAIN_IMG_DIR, gt_df, train_ids,
        mode="train", augment=True,
    )
    val_dataset = FundusDataset(
        src.config.TRAIN_IMG_DIR, gt_df, val_ids,
        mode="val", augment=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False, drop_last=False)

    model = FoveaLocalizationModel(backbone_name=backbone_name, pretrained=True).to(device)
    loss_fn = FoveaLoss()
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
        history.append({
            "epoch": epoch,
            "val_AED": aed,
            "val_median": val_metrics["ED_median"],
            "val_max": val_metrics["ED_max"],
            "val_std": val_metrics["ED_std"],
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"  │ Epoch {epoch:03d} | Val AED: {aed:.4f} | {epoch_time:.1f}s")

        if aed < best_aed:
            best_aed = aed
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_metrics,
                            is_best=True, save_dir=fold_dir)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  │ Early stop @ epoch {epoch} (patience={PATIENCE})")
                break

    total_time = time.time() - t_start
    print(f"  │ Best AED: {best_aed:.4f} @ epoch {best_epoch} ({total_time:.0f}s)")
    print(f"  └────────────────────────────────────────────────────")

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

    return {
        "backbone": backbone_name,
        "fold": fold_k,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "best_aed": best_aed,
        "best_epoch": best_epoch,
        "score": 1.0 / (best_aed + 0.1),
        "total_time": total_time,
    }, history


def plot_convergence(all_histories, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for backbone_name, fold_histories in all_histories.items():
        min_epochs = min(len(h) for h in fold_histories)
        aed_matrix = np.array([h[:min_epochs] for h in fold_histories])
        mean_aed = aed_matrix.mean(axis=0)
        std_aed = aed_matrix.std(axis=0)
        epochs = np.arange(1, min_epochs + 1)

        plt.plot(epochs, mean_aed, label=backbone_name, color=COLORS[backbone_name], linewidth=2)
        plt.fill_between(epochs, mean_aed - std_aed, mean_aed + std_aed,
                         color=COLORS[backbone_name], alpha=0.15)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Validation AED", fontsize=14)
    plt.title("5-Fold CV + Preprocessing: Convergence (mean ± std)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Convergence plot saved to: {save_path}")


def plot_boxplot(results_df, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [results_df[results_df["backbone"] == bb]["best_aed"].values
            for bb in BACKBONES]

    plt.figure(figsize=(8, 5))
    bp = plt.boxplot(data, labels=BACKBONES, patch_artist=True, widths=0.5)

    for patch, color in zip(bp["boxes"], [COLORS[bb] for bb in BACKBONES]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, vals in enumerate(data):
        jitter = np.random.normal(0, 0.04, len(vals))
        plt.scatter(np.full_like(vals, i + 1) + jitter, vals,
                    color=COLORS[BACKBONES[i]], s=50, zorder=5, alpha=0.8)

    plt.xlabel("Backbone", fontsize=14)
    plt.ylabel("Best AED (per fold)", fontsize=14)
    plt.title("5-Fold CV + Preprocessing: AED Distribution", fontsize=16)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Box plot saved to: {save_path}")


def main():
    import src.config
    # Enable full preprocessing pipeline
    src.config.USE_PREPROCESSING = True
    src.config.USE_FOV_MASK = True
    src.config.USE_ILLUM_CORRECTION = True
    src.config.USE_CLAHE = True
    src.config.USE_ASPECT_PRESERVE = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Preprocessing mode: 5-fold CV on {len(ALL_IDS)} training images")

    SCRIPT_DIR = Path(__file__).parent
    CP_DIR = SCRIPT_DIR / "checkpoints" / "5fold_preprocess"
    FIG_DIR = SCRIPT_DIR / "output"
    CP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, val_idx in kf.split(ALL_IDS):
        folds.append(([ALL_IDS[i] for i in train_idx],
                      [ALL_IDS[i] for i in val_idx]))
    print(f"  Folds: {N_FOLDS}, {len(folds[0][0])} train / {len(folds[0][1])} val per fold")

    gt_df = pd.read_excel(src.config.GT_FILE)

    all_results = []
    all_fold_histories = {bb: [] for bb in BACKBONES}
    t_total_start = time.time()

    for backbone_name in BACKBONES:
        from src.dataset import FundusDataset
        FundusDataset.clear_cache()

        n_params = None
        print(f"\n{'=' * 60}")
        print(f"Backbone: {backbone_name}")
        print(f"{'=' * 60}")

        for k, (train_ids, val_ids) in enumerate(folds, 1):
            fold_dir = CP_DIR / backbone_name / f"fold_{k}"
            if (fold_dir / "best_model.pth").exists() and (fold_dir / "history.csv").exists():
                hist_df = pd.read_csv(fold_dir / "history.csv")
                best_row = hist_df.loc[hist_df["val_AED"].idxmin()]
                print(f"  \u23ed\ufe0f  Fold {k} already complete (AED={best_row['val_AED']:.4f}), skipping")
                all_results.append({
                    "backbone": backbone_name, "fold": k,
                    "best_aed": best_row["val_AED"], "best_epoch": int(best_row["epoch"]),
                    "score": 1.0 / (best_row["val_AED"] + 0.1), "total_time": 0,
                })
                all_fold_histories[backbone_name].append(hist_df["val_AED"].tolist())
                continue

            if n_params is None:
                from src.model import FoveaLocalizationModel
                m = FoveaLocalizationModel(backbone_name=backbone_name, pretrained=False)
                n_params = count_parameters(m)
                print(f"  Parameters: {n_params:,}")
                del m

            res, hist = train_one_fold(backbone_name, train_ids, val_ids,
                                       device, CP_DIR / backbone_name, gt_df, k)
            all_results.append(res)
            all_fold_histories[backbone_name].append([h["val_AED"] for h in hist])

    t_total = time.time() - t_total_start
    print(f"\n\n{'=' * 70}")
    print(f"Total training time: {t_total:.0f}s")

    df_results = pd.DataFrame(all_results)

    print(f"\n{'=' * 90}")
    print(f"{'Backbone':<12} {'Fold':>5}  {'AED':>8}  {'Score':>7}  {'Best Ep':>7}  {'Time':>7}")
    print(f"{'-'*90}")
    for _, row in df_results.iterrows():
        print(f"{row['backbone']:<12} {row['fold']:>5d}  "
              f"{row['best_aed']:>8.4f}  {row['score']:>7.2f}  "
              f"{row['best_epoch']:>7d}  {row['total_time']:>7.0f}s")

    print(f"\n{'=' * 70}")
    print(f"{'Backbone':<12} {'AED_mean':>10} {'AED_std':>9} {'AED_min':>9} "
          f"{'Score_m':>9} {'Score_std':>9}")
    print(f"{'-'*60}")
    agg = df_results.groupby("backbone")["best_aed"].agg(["mean", "std", "min", "max"])
    score_agg = df_results.groupby("backbone")["score"].agg(["mean", "std"])
    for bb in BACKBONES:
        print(f"{bb:<12}  {agg.loc[bb,'mean']:>8.4f}   {agg.loc[bb,'std']:>7.4f}  "
              f"{agg.loc[bb,'min']:>8.4f}   {score_agg.loc[bb,'mean']:>7.2f}   "
              f"{score_agg.loc[bb,'std']:>7.2f}")

    df_results.to_csv(CP_DIR / "5fold_summary.csv", index=False)

    agg_df = pd.DataFrame({
        "backbone": BACKBONES,
        "AED_mean": agg["mean"].values,
        "AED_std": agg["std"].values,
        "AED_min": agg["min"].values,
        "Score_mean": score_agg["mean"].values,
        "Score_std": score_agg["std"].values,
    })
    agg_df.to_csv(CP_DIR / "5fold_aggregated.csv", index=False)
    print(f"\n  Results saved to: {CP_DIR}")

    best_bb = agg_df.loc[agg_df["AED_mean"].idxmin(), "backbone"]
    best_mean = agg_df.loc[agg_df["AED_mean"].idxmin(), "AED_mean"]
    print(f"\n  Best backbone by mean AED: {best_bb} (AED={best_mean:.4f})")

    plot_convergence(all_fold_histories, FIG_DIR / "5fold_preprocess_convergence.png")
    plot_boxplot(df_results, FIG_DIR / "5fold_preprocess_boxplot.png")
    print(f"\n  All plots saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
