"""
方案演进可视化：在同一张图片上对比各实验方案的定位效果

逐行展示 4 张代表性图片（ID: 1, 38, 73, 92）
逐列展示 4 个演进方案 + 1 列 GT 参考：
  GT参考 → Exp-01(基线) → Exp-03(+预处理) → Exp-05(热图) → Exp-07(集成+TTA)

运行:
    cd d:\OneDrive\桌面\最终\新工作区
    python visualize_progression.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import cv2
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(max(1, os.cpu_count() // 2))

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from src.config import *
from src.model import FoveaLocalizationModel, HeatmapLocModel, soft_argmax
from src.dataset import _fundus_preprocess, _aspect_preserve_resize_pad
from train_baseline_origin import OriginBaselineModel
from src.utils import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
SAMPLE_IDS = [1, 32, 46, 85]  # 改进最大的 4 张图（基线差→最优好）


# ========================================================================
#  Helper: load and preprocess a single image
# ========================================================================

def load_and_preprocess(pid, use_preprocessing, use_intermediate):
    """Load image and transform to model input.

    Returns:
        input_tensor: [1, 3, 512, 512] float32 on DEVICE
        raw_img: [H,W,3] uint8 original raw image (for display)
        gt_orig: (x, y) original pixel GT coords
        params: dict for coordinate denormalization
    """
    gt_df = pd.read_excel(GT_FILE)
    row = gt_df[gt_df["data"] == pid].iloc[0]
    gt_x, gt_y = float(row["Fovea_X"]), float(row["Fovea_Y"])

    img_path = TRAIN_IMG_DIR / f"{pid:04d}.jpg"
    img_bytes = np.fromfile(img_path, dtype=np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raw_img = image.copy()
    orig_h, orig_w = image.shape[:2]

    # Intermediate resize
    inter_scale = 1.0
    if use_preprocessing and use_intermediate:
        max_dim = max(orig_h, orig_w)
        if max_dim > PREPROCESS_INTER_SIZE:
            inter_scale = PREPROCESS_INTER_SIZE / max_dim
            new_w = round(orig_w * inter_scale)
            new_h = round(orig_h * inter_scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pre_h, pre_w = image.shape[:2]

    # Preprocessing pipeline (or skip)
    if use_preprocessing:
        image = _fundus_preprocess(image)

    # Aspect-preserve resize + pad to 512
    image, (pad_x, pad_y, scale) = _aspect_preserve_resize_pad(
        image, target_size=IMG_SIZE)

    # Normalize coord to [0, 1] in 512-space (for GT)
    coord_norm = np.array([
        (gt_x * inter_scale / pre_w * pre_w * scale + pad_x) / IMG_SIZE,
        (gt_y * inter_scale / pre_h * pre_h * scale + pad_y) / IMG_SIZE,
    ], dtype=np.float32)
    # Simplify: same formula as dataset
    coord_norm = np.array([
        (gt_x * inter_scale / pre_w * pre_w * scale + pad_x) / IMG_SIZE,
        (gt_y * inter_scale / pre_h * pre_h * scale + pad_y) / IMG_SIZE,
    ], dtype=np.float32)
    # Actually the coord normalization formula from the dataset:
    # coord[0] = (coord[0] * pre_w * scale + pad_x) / IMG_SIZE
    # where coord[0] = x_gt * inter_scale / pre_w
    # So: coord_norm[0] = (x_gt * inter_scale / pre_w * pre_w * scale + pad_x) / IMG_SIZE
    #                      = (x_gt * inter_scale * scale + pad_x) / IMG_SIZE
    coord_norm = np.array([
        (gt_x * inter_scale * scale + pad_x) / IMG_SIZE,
        (gt_y * inter_scale * scale + pad_y) / IMG_SIZE,
    ], dtype=np.float32)

    # Pixel normalization (ImageNet)
    input_tensor = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_tensor = (input_tensor - mean) / std
    input_tensor = input_tensor.transpose(2, 0, 1)  # HWC → CHW
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)

    params = {
        "inter_scale": inter_scale,
        "pad_x": pad_x, "pad_y": pad_y,
        "scale": scale,
        "pre_w": pre_w, "pre_h": pre_h,
        "orig_w": orig_w, "orig_h": orig_h,
    }

    return input_tensor, raw_img, (gt_x, gt_y), coord_norm, params


def denormalize_coord(norm_coord, params):
    """Convert [0,1] normalized 512-coord back to original pixel space."""
    px = (norm_coord[0] * IMG_SIZE - params["pad_x"]) / (params["scale"] * params["inter_scale"])
    py = (norm_coord[1] * IMG_SIZE - params["pad_y"]) / (params["scale"] * params["inter_scale"])
    return px, py


# ========================================================================
#  Model inference
# ========================================================================

@torch.no_grad()
def predict_coord_reg(model, input_tensor):
    """Coordinate regression model: FoveaLocalizationModel."""
    coord = model(input_tensor)  # [B, 2]
    return coord[0].cpu().numpy()  # [2]


@torch.no_grad()
def predict_heatmap(model, input_tensor, use_tta=False):
    """Heatmap model: HeatmapLocModel, optional TTA."""
    coords, _ = model(input_tensor)
    if use_tta:
        input_flip = input_tensor.flip(-1)
        coords_flip, _ = model(input_flip)
        coords_flip[:, 0] = 1.0 - coords_flip[:, 0]
        coords = (coords + coords_flip) / 2.0
    return coords[0].cpu().numpy()


@torch.no_grad()
def predict_ensemble(models, input_tensor):
    """Ensemble of heatmap models with TTA."""
    all_preds = []
    for m in models:
        pred = predict_heatmap(m, input_tensor, use_tta=True)
        all_preds.append(pred)
    return np.mean(all_preds, axis=0)


@torch.no_grad()
def predict_baseline(model, raw_img):
    """Official baseline model: ResNet50+3FC, 256×256 direct resize, /255.0 norm."""
    # Resize to 256×256 (stretch)
    img = cv2.resize(raw_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    # /255.0 normalization
    img = img.astype(np.float32) / 255.0
    # HWC → CHW → [1, 3, 256, 256]
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    coord = model(img)
    return coord[0].cpu().numpy()  # [2] normalized by orig W/H


def load_baseline_model(ckpt_path):
    model = OriginBaselineModel().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ========================================================================
#  Models loading
# ========================================================================

def load_fovea_model(backbone, ckpt_path):
    model = FoveaLocalizationModel(backbone_name=backbone, pretrained=False).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_heatmap_model(backbone, ckpt_path):
    model = HeatmapLocModel(backbone_name=backbone, pretrained=False).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ========================================================================
#  Plotting
# ========================================================================

def compute_aed(pred_orig, gt_orig, orig_h, orig_w):
    """Compute AED in original pixel space."""
    ed = np.sqrt((pred_orig[0] - gt_orig[0])**2 + (pred_orig[1] - gt_orig[1])**2)
    aed = ed / np.sqrt(orig_w * orig_h)
    return aed, ed


def draw_comparison(sample_ids, output_path):
    """Main visualization: 4 images × 5 schemes grid."""

    SCRIPT_DIR = Path(__file__).parent

    # ---- Load models ----
    print("[1/4] Loading models...")

    # Exp-00: Official baseline (ResNet50 + 3FC)
    model_exp00 = load_baseline_model(
        SCRIPT_DIR / "output" / "checkpoints" / "baseline_origin" / "best_model.pth")
    print("  Loaded Exp-00: Official baseline")

    # Exp-01: ResNet18 coordinate regression (baseline, no preprocessing)
    model_exp01 = load_fovea_model(
        "resnet18", SCRIPT_DIR / "checkpoints" / "comparison" / "resnet18" / "best_model.pth")
    print("  Loaded Exp-01: ResNet18 coordinate regression")

    # Exp-05: ResNet34 heatmap
    model_exp05 = load_heatmap_model(
        "resnet34", SCRIPT_DIR / "checkpoints" / "5fold_heatmap" / "resnet34" / "resnet34" / "fold_1" / "best_model.pth")
    print("  Loaded Exp-05: ResNet34 heatmap")

    # Exp-07: Ensemble of 5 ResNet34 heatmaps + TTA
    ensemble_models = []
    for k in range(1, 6):
        cp = SCRIPT_DIR / "checkpoints" / "5fold_heatmap" / "resnet34" / "resnet34" / f"fold_{k}" / "best_model.pth"
        m = load_heatmap_model("resnet34", cp)
        ensemble_models.append(m)
    print(f"  Loaded Exp-07: Ensemble ({len(ensemble_models)} models)")

    schemes = [
        ("Exp-00 官方基线\n(ResNet50+3FC)", model_exp00, "baseline", None),
        ("Exp-01 改进\n(ResNet18 坐标回归)", model_exp01, "coord_reg", False),
        ("Exp-05 热图\n(ResNet34)", model_exp05, "heatmap", True),
        ("Exp-07 集成+TTA\n(当前最优)", ensemble_models, "ensemble", True),
    ]

    # ---- Inference on all images ----
    print("[2/4] Running inference on sample images...")
    results = []  # list of (pid, raw_img, gt_orig, [(scheme_name, pred_orig, aed, ed_px), ...])

    for pid in sample_ids:
        print(f"  Processing ID {pid:04d}...")

        # Need two versions: with and without preprocessing
        # For each scheme, use the appropriate preprocessing mode
        inputs = {}

        # Exp-01: no preprocessing, no intermediate resize
        inp_nopp, raw, gt_orig, gt_norm, params_nopp = load_and_preprocess(
            pid, use_preprocessing=False, use_intermediate=False)
        inputs["nopp"] = (inp_nopp, params_nopp)

        # Exp-03/05/07: with preprocessing, with intermediate resize
        inp_pp, raw, gt_orig, gt_norm, params_pp = load_and_preprocess(
            pid, use_preprocessing=True, use_intermediate=True)
        inputs["pp"] = (inp_pp, params_pp)

        scheme_results = []
        for name, model, model_type, use_pp in schemes:
            if model_type == "baseline":
                # Baseline: different input pipeline (256×256 direct resize, /255.0)
                pred_norm = predict_baseline(model, raw)
                # Baseline coords are normalized by orig W/H
                pred_orig = (pred_norm[0] * raw.shape[1], pred_norm[1] * raw.shape[0])
            else:
                inp, params = inputs["pp"] if use_pp else inputs["nopp"]

                if model_type == "coord_reg":
                    pred_norm = predict_coord_reg(model, inp)
                elif model_type == "heatmap":
                    pred_norm = predict_heatmap(model, inp, use_tta=False)
                elif model_type == "ensemble":
                    pred_norm = predict_ensemble(model, inp)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                pred_orig = denormalize_coord(pred_norm, params)

            aed, ed_px = compute_aed(pred_orig, gt_orig, raw.shape[0], raw.shape[1])
            scheme_results.append((name.split("\n")[0], pred_orig, aed, ed_px))

        results.append((pid, raw, gt_orig, scheme_results, params_nopp))

    # ---- Plot ----
    print("[3/4] Generating figure...")

    n_rows = len(sample_ids)
    n_cols = 1 + len(schemes)  # GT reference + 4 schemes
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.8, n_rows * 3.8),
                             constrained_layout=True)

    scheme_labels = ["GT 参考"] + [s[0] for s in schemes]

    # Colors
    gt_color = "#2ecc71"  # green
    pred_color = "#e74c3c"  # red

    for row_idx, (pid, raw_img, gt_orig, scheme_results, params) in enumerate(results):
        orig_h, orig_w = raw_img.shape[:2]

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.clear()

            # Background: show raw image for ALL columns
            ax.imshow(raw_img)

            # GT marker in ALL columns
            gt_plot_x, gt_plot_y = gt_orig
            gt_circle = Circle((gt_plot_x, gt_plot_y), radius=12,
                               fill=False, edgecolor=gt_color, linewidth=2.5,
                               label="GT")
            ax.add_patch(gt_circle)
            ax.plot(gt_plot_x, gt_plot_y, marker="+", color=gt_color,
                    markersize=10, markeredgewidth=2.5)

            if col_idx == 0:
                # Column 0: GT reference only (full image view)
                ax.set_title("GT 参考", fontsize=11, fontweight="bold", pad=6)
                ax.set_xlim(0, orig_w)
                ax.set_ylim(orig_h, 0)
            else:
                # Columns 1-4: prediction (zoomed view around fovea)
                scheme_label, pred_orig, aed, ed_px = scheme_results[col_idx - 1]

                pred_circle = Circle((pred_orig[0], pred_orig[1]), radius=12,
                                      fill=False, edgecolor=pred_color, linewidth=2.5,
                                      label="Pred")
                ax.add_patch(pred_circle)
                ax.plot(pred_orig[0], pred_orig[1], marker="x", color=pred_color,
                        markersize=10, markeredgewidth=2.5)

                # Error line from GT to pred
                ax.plot([gt_plot_x, pred_orig[0]], [gt_plot_y, pred_orig[1]],
                        color="#f39c12", linewidth=1.5, linestyle="--", alpha=0.8)

                # Zoomed region around GT
                zoom_radius = max(int(max(orig_h, orig_w) * 0.08), 60)
                ax.set_xlim(gt_plot_x - zoom_radius, gt_plot_x + zoom_radius)
                ax.set_ylim(gt_plot_y + zoom_radius, gt_plot_y - zoom_radius)

                # Title with error
                title_text = f"{scheme_label}\nAED={aed:.4f}  (Δ{ed_px:.0f}px)"
                ax.set_title(title_text, fontsize=9, fontweight="bold", pad=6)

            ax.set_xticks([])
            ax.set_yticks([])

            # Row label
            if col_idx == 0:
                ax.set_ylabel(f"ID {pid:04d}\n({orig_w}×{orig_h})",
                              fontsize=10, fontweight="bold")

    # Global legend
    handles = [
        plt.Line2D([], [], color=gt_color, marker="+", linewidth=0,
                   markersize=8, label="GT (黄斑中心凹)"),
        plt.Line2D([], [], color=pred_color, marker="x", linewidth=0,
                   markersize=8, label="Pred (预测)"),
        plt.Line2D([], [], color="#f39c12", linestyle="--", linewidth=1.5,
                   label="误差 (像素)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    # Super title
    fig.suptitle("实验方案演进 — 黄斑中心凹定位精度对比",
                 fontsize=14, fontweight="bold", y=1.01)

    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Figure saved to: {output_path}")


if __name__ == "__main__":
    output_path = Path(__file__).parent / "output" / "progression_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_comparison(SAMPLE_IDS, output_path)
    print("\nDone! Check output/progression_comparison.png")
