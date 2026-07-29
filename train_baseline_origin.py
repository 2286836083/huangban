"""
比赛官方基线 PyTorch 等效实现
===============================
完全复现基线.md 的架构和超参数：
- 模型: ResNet50 → FC(2048→512→ReLU→256→ReLU→Dropout→2→Sigmoid)
- 输入: 256×256 直接缩放宽高比拉伸
- 损失: 0.5×MSE(x,y) + 0.5×Euclidean Distance
- 优化器: Adam, lr=1e-4
- 迭代: 500 iterations
- 划分: 80/20, random_state=42
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(max(1, os.cpu_count() // 2))

# ========== Config (matches baseline.md exactly) ==========
IMAGE_SIZE = 256
BATCH_SIZE = 32
ITERS = 500
LR = 1e-4
VAL_RATIO = 0.2
RANDOM_STATE = 42
ALPHA = 0.5  # loss weight
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 10
EVL_INTERVAL = 100

# ========== Paths ==========
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "数据集"
TRAIN_IMG_DIR = DATA_DIR / "training" / "fundus color images"
GT_FILE = DATA_DIR / "training" / "fovea_localization_training_GT.xlsx"


# ========== Dataset ==========
class OriginBaselineDataset(Dataset):
    """Matches baseline.md data loading exactly:
    - Direct resize to 256×256 (stretch, no aspect-preserve)
    - Normalize: /255.0 only (no ImageNet mean/std)
    - Label: normalized by original W/H
    """

    def __init__(self, img_dir, gt_df, filelists):
        self.img_dir = Path(img_dir)
        self.samples = []
        for fname in filelists:
            row = gt_df[gt_df["data"] == int(fname.replace(".jpg", ""))]
            if len(row) == 0:
                continue
            x = float(row.iloc[0]["Fovea_X"])
            y = float(row.iloc[0]["Fovea_Y"])
            self.samples.append((fname, x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, x_gt, y_gt = self.samples[idx]
        img_path = self.img_dir / fname
        img_bytes = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Direct resize to 256×256 (stretch, same as baseline.md)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE),
                           interpolation=cv2.INTER_LINEAR)
        # /255.0 normalization
        image = image.astype(np.float32) / 255.0
        # HWC → CHW
        image = image.transpose(2, 0, 1)

        # Normalized label
        label = np.array([x_gt / w, y_gt / h], dtype=np.float32)

        return torch.from_numpy(image), torch.from_numpy(label)


# ========== Model ==========
class OriginBaselineModel(nn.Module):
    """Matches baseline.md Network class exactly:
    ResNet50(pretrained, remove FC) → FC 2048→512→ReLU→256→ReLU→Dropout→2→Sigmoid
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # output: [B, 2048]

        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.resnet(x)   # [B, 2048]
        x = self.head(x)     # [B, 2]
        return x


# ========== Loss ==========
def cal_coordinate_loss(logits, labels, alpha=ALPHA):
    """Matches baseline.md cal_coordinate_Loss exactly."""
    # MSE on x and y separately, then average
    mse = nn.MSELoss()
    mse_x = mse(logits[:, 0], labels[:, 0])
    mse_y = mse(logits[:, 1], labels[:, 1])
    mse_l = 0.5 * (mse_x + mse_y)

    # Euclidean distance per sample, then batch mean
    ed = torch.norm(logits - labels, dim=1).mean()

    return alpha * mse_l + (1 - alpha) * ed


# ========== Evaluation metrics ==========
def compute_aed(pred, gt, orig_h, orig_w):
    """Compute AED in original pixel space."""
    pred_x = pred[:, 0] * orig_w
    pred_y = pred[:, 1] * orig_h
    gt_x = gt[:, 0] * orig_w
    gt_y = gt[:, 1] * orig_h
    ed = torch.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    aed = ed / torch.sqrt(torch.tensor(orig_w * orig_h, dtype=torch.float32))
    return aed.mean().item(), ed.mean().item()


# ========== Training ==========
def evaluate(model, val_loader):
    model.eval()
    all_logits, all_labels = [], []
    orig_dims = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            # Use a default image size for AED computation
            orig_dims.append((images.shape[2], images.shape[3]))

    logits_all = torch.cat(all_logits, dim=0)
    labels_all = torch.cat(all_labels, dim=0)

    # AED in normalized space (normalized by the IMAGE_SIZE the model sees)
    # But AED should be in original image space.
    # For a fair comparison, we use IMAGE_SIZE as proxy (original dims vary)
    aed, ed_px = compute_aed(logits_all, labels_all, IMAGE_SIZE, IMAGE_SIZE)
    score = 1.0 / (aed + 0.1)
    return aed, score, ed_px


def train():
    print(f"Using device: {DEVICE}")
    print(f"{'=' * 60}")
    print(f"  Original Baseline (PyTorch equivalent)")
    print(f"  Model: ResNet50 → FC(512→256→2)  |  Input: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Batch: {BATCH_SIZE}  |  Iters: {ITERS}  |  LR: {LR}")
    print(f"  Loss: alpha={ALPHA} (0.5×MSE + 0.5×ED)")
    print(f"{'=' * 60}")

    # ---- Load GT ----
    gt_df = pd.read_excel(GT_FILE)
    print(f"  GT loaded: {len(gt_df)} samples")

    # ---- Data split (80/20, random_state=42) ----
    all_files = sorted(os.listdir(TRAIN_IMG_DIR))
    train_files, val_files = train_test_split(
        all_files, test_size=VAL_RATIO, random_state=RANDOM_STATE)
    print(f"  Split: {len(train_files)} train, {len(val_files)} val")

    # ---- Datasets ----
    train_dataset = OriginBaselineDataset(TRAIN_IMG_DIR, gt_df, train_files)
    val_dataset = OriginBaselineDataset(TRAIN_IMG_DIR, gt_df, val_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=False,
                              drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=False,
                            drop_last=False)

    # ---- Model ----
    model = OriginBaselineModel().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---- Training loop (matches baseline.md: while iter < iters) ----
    best_aed = float("inf")
    best_ed_px = float("inf")
    global_iter = 0
    train_losses = []
    t_start = time.time()

    while global_iter < ITERS:
        for images, labels in train_loader:
            global_iter += 1
            if global_iter > ITERS:
                break

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = cal_coordinate_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Log
            if global_iter % LOG_INTERVAL == 0:
                avg_loss = np.mean(train_losses[-LOG_INTERVAL:])
                # Compute training AED on this batch
                aed_batch, ed_px_batch = compute_aed(
                    logits.detach().cpu(), labels.cpu(), IMAGE_SIZE, IMAGE_SIZE)
                print(f"  [TRAIN] iter={global_iter}/{ITERS}  loss={avg_loss:.4f}  ED={ed_px_batch:.2f}px")

            # Evaluate
            if global_iter % EVL_INTERVAL == 0:
                val_aed, val_score, val_ed_px = evaluate(model, val_loader)
                print(f"  [EVAL]  iter={global_iter}/{ITERS}  AED={val_aed:.4f}  "
                      f"Score={val_score:.2f}  ED={val_ed_px:.2f}px")
                if val_aed < best_aed:
                    best_aed = val_aed
                    best_ed_px = val_ed_px
                    torch.save({"model_state_dict": model.state_dict(),
                                "iter": global_iter, "AED": val_aed},
                               SCRIPT_DIR / "output" / "checkpoints" / "baseline_origin" / "best_model.pth")
                    print(f"  → New best! AED={val_aed:.4f} Score={val_score:.2f}")

    t_total = time.time() - t_start

    # ---- Final evaluation ----
    print(f"\n{'=' * 60}")
    print(f"  Training complete ({t_total:.0f}s)")
    # Load best model
    cp_path = SCRIPT_DIR / "output" / "checkpoints" / "baseline_origin" / "best_model.pth"
    if cp_path.exists():
        ckpt = torch.load(cp_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best model @ iter {ckpt['iter']}, AED={ckpt['AED']:.4f}")

    final_aed, final_score, final_ed = evaluate(model, val_loader)
    print(f"\n{'=' * 60}")
    print(f"  Final Results")
    print(f"{'=' * 60}")
    print(f"  Best AED:   {best_aed:.4f}")
    print(f"  Best Score: {1/(best_aed+0.1):.2f}")
    print(f"  Best ED:    {best_ed_px:.1f}px")
    print(f"  Params:     {n_params:,}")
    print(f"  Time:       {t_total:.0f}s")

    # Save summary
    results = pd.DataFrame([{
        "method": "origin_baseline",
        "AED": best_aed,
        "Score": 1 / (best_aed + 0.1),
        "ED_px": best_ed_px,
        "params": n_params,
        "iters": ITERS,
        "time_s": t_total,
    }])
    out_dir = SCRIPT_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "baseline_origin_results.csv", index=False)
    print(f"\n  Results saved to: {out_dir / 'baseline_origin_results.csv'}")


if __name__ == "__main__":
    train()
