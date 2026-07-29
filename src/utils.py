"""Utility functions: metrics, visualization, etc."""

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .config import OUTPUT_DIR, VAL_IDS


def euclidean_distance(pred, target):
    """Compute Euclidean distance between predicted and ground truth coordinates.

    Args:
        pred: [N, 2] or [2]
        target: [N, 2] or [2]

    Returns:
        distances: [N] or scalar
    """
    if isinstance(pred, torch.Tensor):
        return torch.sqrt(((pred - target) ** 2).sum(dim=-1))
    return np.sqrt(((pred - target) ** 2).sum(axis=-1))


def compute_metrics(pred_coords, target_coords):
    """Compute comprehensive evaluation metrics.

    Args:
        pred_coords: [N, 2] in normalized [0, 1] (can be converted to pixel)
        target_coords: [N, 2]

    Returns:
        dict of metrics
    """
    eds = euclidean_distance(pred_coords, target_coords)

    if isinstance(eds, torch.Tensor):
        eds = eds.cpu().numpy()

    return {
        "ED_mean": float(np.mean(eds)),
        "ED_median": float(np.median(eds)),
        "ED_std": float(np.std(eds)),
        "ED_min": float(np.min(eds)),
        "ED_max": float(np.max(eds)),
    }


def visualize_prediction(image, pred_coord, gt_coord, heatmap=None, ed=None, save_path=None):
    """Visualize prediction results on image.

    Args:
        image: [H, W, 3] RGB image (numpy, 0-255 or 0-1)
        pred_coord: (x, y) in pixel coordinates (relative to image size)
        gt_coord: (x, y) in pixel coordinates (relative to image size)
        heatmap: [H_hm, W_hm] heatmap or None (will be resized to match image)
        ed: Euclidean distance (normalized), displayed as text overlay
        save_path: if provided, save figure
    """
    h, w = image.shape[:2]

    fig, axes = plt.subplots(1, 2 if heatmap is None else 3, figsize=(15, 5))

    ax = axes[0]
    ax.imshow(image)
    ax.plot(gt_coord[0], gt_coord[1], "g+", markersize=15, label="GT")
    ax.plot(pred_coord[0], pred_coord[1], "rx", markersize=15, label="Pred")
    ax.legend()
    title = "Prediction vs GT"
    if ed is not None:
        title += f"\nED (norm) = {ed:.4f}"
    ax.set_title(title)
    ax.axis("off")

    if heatmap is not None:
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        ax = axes[1]
        im = ax.imshow(heatmap_resized, cmap="jet", extent=[0, w, h, 0])
        ax.plot(gt_coord[0], gt_coord[1], "g+", markersize=15, label="GT")
        ax.plot(pred_coord[0], pred_coord[1], "rx", markersize=15, label="Pred")
        ax.legend()
        ax.set_title("Heatmap")
        ax.axis("off")
        plt.colorbar(im, ax=ax)

        ax = axes[2]
        # Normalize image to [0,1] for overlay if needed
        if image.max() > 1.0:
            img_display = image / 255.0
        else:
            img_display = image
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        overlay[:, :, 0] = heatmap_resized  # red channel for heatmap
        blended = img_display * 0.6 + overlay * 0.4
        blended = np.clip(blended, 0, 1)
        ax.imshow(blended)
        ax.plot(gt_coord[0], gt_coord[1], "g+", markersize=15, label="GT")
        ax.plot(pred_coord[0], pred_coord[1], "rx", markersize=15, label="Pred")
        ax.legend()
        ax.set_title("Overlay (heatmap in red)")
        ax.axis("off")
    else:
        ax = axes[1]
        ax.hist(euclidean_distance(np.array(pred_coord), np.array(gt_coord)), bins=20)
        ax.set_xlabel("Euclidean Distance (pixels)")
        ax.set_ylabel("Count")
        ax.set_title("Error Distribution")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_batch_results(
    images, pred_coords_pixel, gt_coords_pixel, heatmaps=None,
    save_dir="visualizations", prefix="val"
):
    """Visualize a batch of results and save individual figures."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(len(images), 8)):
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy().transpose(1, 2, 0)
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img * std + mean) * 255
            img = np.clip(img, 0, 255).astype(np.uint8)

        h, w = img.shape[:2]
        gt = (gt_coords_pixel[i] if isinstance(gt_coords_pixel, list)
              else gt_coords_pixel[i].cpu().numpy())
        pred = (pred_coords_pixel[i] if isinstance(pred_coords_pixel, list)
                else pred_coords_pixel[i].cpu().numpy())

        hm = None
        if heatmaps is not None:
            hm = heatmaps[i]
            if isinstance(hm, torch.Tensor):
                hm = hm.cpu().numpy().squeeze()
            hm = cv2.resize(hm, (w, h))

        save_path = save_dir / f"{prefix}_sample_{i}.png"
        visualize_prediction(img, pred, gt, heatmap=hm, save_path=save_path)


def save_checkpoint(model, optimizer, epoch, metrics, is_best=False, save_dir=None):
    """Save model checkpoint."""
    if save_dir is None:
        save_dir = Path(OUTPUT_DIR) / "checkpoints"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    filename = f"epoch_{epoch:03d}_ed_{metrics.get('ED_mean', 0):.4f}.pth"
    if is_best:
        filename = "best_model.pth"
    torch.save(checkpoint, str(save_dir / filename))
    return str(save_dir / filename)
