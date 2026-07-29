"""Dataset and data loading for Fovea Localization."""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import *


# ========================================================================
#  Preprocessing Pipeline
# ========================================================================

def _ensure_odd(k):
    """Ensure kernel size is odd."""
    return k + 1 if k % 2 == 0 else k


def _detect_fov_mask(image):
    """Detect circular FOV (field of view) mask from fundus image.

    Args:
        image: [H, W, 3] RGB uint8 array

    Returns:
        mask: [H, W] uint8 binary mask (255 = FOV, 0 = background)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Threshold to separate bright fundus from dark background
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # Keep only the largest connected region (the FOV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.ones_like(gray, dtype=np.uint8) * 255  # fallback: all ones
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    # Fill and smooth the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = (mask > 127).astype(np.uint8) * 255
    return mask


def _illumination_normalize(image, mask=None):
    """Correct non-uniform illumination using large-kernel Gaussian blur.

    Estimates a smooth illumination field (vignetting pattern) and divides
    the image by it, removing center-bright / edge-dark artifacts.

    Args:
        image: [H, W, 3] RGB uint8 array
        mask: optional [H, W] binary mask to zero out background regions

    Returns:
        corrected: [H, W, 3] RGB uint8 array
    """
    h, w = image.shape[:2]
    kernel_size = _ensure_odd(int(min(h, w) / ILLUM_KERNEL_RATIO))
    kernel_size = max(kernel_size, 15)  # minimum kernel size

    result = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        # Estimate illumination field via large-kernel blur
        illum = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)
        # Avoid division by very small values
        illum_mean = illum[mask > 0].mean() if mask is not None and mask.any() else np.mean(illum)
        illum = np.where(illum < 1, illum_mean * 0.5, illum)
        # Correct: divide by illumination, then rescale to retain brightness
        corrected = channel / illum * illum_mean
        if mask is not None:
            corrected = corrected * (mask.astype(np.float32) / 255.0)
        result[:, :, c] = np.clip(corrected, 0, 255)

    return result.astype(np.uint8)


def apply_clahe(image, clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_SIZE):
    """Apply CLAHE contrast enhancement to RGB image.

    Operates on the L-channel of LAB color space to enhance contrast
    without introducing color artifacts.

    Args:
        image: [H, W, 3] RGB uint8 array
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile grid size

    Returns:
        enhanced: [H, W, 3] RGB uint8 array
    """
    if not USE_CLAHE:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_planes = lab.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab_planes[:, :, 0] = clahe.apply(lab_planes[:, :, 0])
    enhanced = cv2.cvtColor(lab_planes, cv2.COLOR_LAB2RGB)
    return enhanced


def _aspect_preserve_resize_pad(image, target_size=IMG_SIZE):
    """Resize image preserving aspect ratio and pad to square.

    The shorter side is scaled to target_size; the longer side is scaled
    proportionally and centered with symmetric padding.

    Args:
        image: [H, W, 3] RGB uint8 array
        target_size: target square size (default: IMG_SIZE)

    Returns:
        padded: [target_size, target_size, 3] RGB uint8 array
        (pad_x, pad_y, scale): transformation info for coordinate adjustment
            - pad_x: horizontal padding on each side (pixels)
            - pad_y: vertical padding on each side (pixels)
            - scale:  target_size / min(orig_h, orig_w)
    """
    if not USE_ASPECT_PRESERVE:
        # Direct resize fallback (stretches image)
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return resized, (0, 0, target_size / max(image.shape[0], image.shape[1]))

    h, w = image.shape[:2]

    # Skip if already square
    if h == w:
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return resized, (0, 0, target_size / h)

    # Scale so that the LONGER side = target_size (image fits INSIDE target square)
    scale = target_size / max(h, w)
    new_w = round(w * scale)
    new_h = round(h * scale)
    # Ensure we don't exceed target
    new_w = min(new_w, target_size)
    new_h = min(new_h, target_size)

    # Resize preserving aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    padded = np.full((target_size, target_size, 3), 0, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return padded, (pad_x, pad_y, scale)


def _fundus_preprocess(image):
    """Apply full preprocessing pipeline to a fundus image.

    Pipeline order:
      1. FOV mask (zero out background)
      2. Illumination normalization (remove vignetting)
      3. CLAHE (contrast enhancement)
      (Aspect-ratio preserve + pad is handled separately in __getitem__)

    Args:
        image: [H, W, 3] RGB uint8 array

    Returns:
        processed: [H, W, 3] RGB uint8 array (same size as input)
    """
    if not USE_PREPROCESSING:
        return image

    # 1. FOV mask
    mask = None
    if USE_FOV_MASK:
        mask = _detect_fov_mask(image)
        image = cv2.bitwise_and(image, image, mask=mask)

    # 2. Illumination normalization
    if USE_ILLUM_CORRECTION:
        image = _illumination_normalize(image, mask=mask)

    # 3. CLAHE
    if USE_CLAHE:
        image = apply_clahe(image)

    return image


# ========================================================================
#  Augmentation
# ========================================================================

def apply_geometric_augmentation(image, coord_norm):
    """Apply geometric augmentations to image and update coordinates.

    NOTE: image is expected to already be [IMG_SIZE, IMG_SIZE].
    The initial resize from the original version has been removed — it
    now assumes the preprocessing pipeline has already produced a
    square image.

    Args:
        image: [IMG_SIZE, IMG_SIZE, 3] RGB uint8 array
        coord_norm: [x, y] normalized coordinates in [0, 1]

    Returns:
        aug_image: [IMG_SIZE, IMG_SIZE, 3] uint8
        aug_coord: [x, y] updated normalized coordinates
    """
    img = image.copy()
    coord = coord_norm.copy()

    # 1. Horizontal flip (50%)
    if np.random.random() < 0.5:
        img = cv2.flip(img, 1)
        coord[0] = 1.0 - coord[0]

    # 2. Rotation (50%)
    if np.random.random() < 0.5:
        angle = np.random.uniform(-ROTATION_DEGREES, ROTATION_DEGREES)
        center = (IMG_SIZE / 2, IMG_SIZE / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rot_mat, (IMG_SIZE, IMG_SIZE), borderValue=(0, 0, 0))

        # Update coordinate
        cx_px = coord[0] * IMG_SIZE
        cy_px = coord[1] * IMG_SIZE
        new_cx = rot_mat[0, 0] * cx_px + rot_mat[0, 1] * cy_px + rot_mat[0, 2]
        new_cy = rot_mat[1, 0] * cx_px + rot_mat[1, 1] * cy_px + rot_mat[1, 2]
        new_cx = np.clip(new_cx, 0, IMG_SIZE)
        new_cy = np.clip(new_cy, 0, IMG_SIZE)
        coord[0] = new_cx / IMG_SIZE
        coord[1] = new_cy / IMG_SIZE

    # 3. Brightness/Contrast (30%)
    if np.random.random() < 0.3:
        alpha = 1.0 + np.random.uniform(-CONTRAST_LIMIT, CONTRAST_LIMIT)
        beta = np.random.uniform(-BRIGHTNESS_LIMIT * 255, BRIGHTNESS_LIMIT * 255)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img, coord


# ========================================================================
#  Dataset
# ========================================================================

class FundusDataset(Dataset):
    """Fovea Localization Dataset with preprocessing pipeline.

    Train/val returns:
        image: [3, IMG_SIZE, IMG_SIZE] float32 tensor (normalized)
        coord: [2] float32 tensor (normalized x, y)

    Test returns:
        image: [3, IMG_SIZE, IMG_SIZE] float32 tensor (normalized)
        fname: str
        orig_h: int (original image height)
        orig_w: int (original image width)
        pad_x: int (padding applied on x-axis)
        pad_y: int (padding applied on y-axis)
        scale: float (scale factor = IMG_SIZE / min(orig_h, orig_w))
    """

    # Class-level image cache: avoids repeated JPEG decode every epoch
    _image_cache = {}
    # Preprocess cache: stores (preprocessed_image, inter_scale, pre_h, pre_w)
    # Avoiding re-running FOV + Illumination + CLAHE every epoch
    _preproc_cache = {}

    @classmethod
    def clear_cache(cls):
        cls._image_cache.clear()
        cls._preproc_cache.clear()

    def __init__(self, img_dir, gt_df=None, ids=None, mode="train", augment=False):
        self.img_dir = img_dir
        self.mode = mode
        self.augment = augment and (mode == "train")

        if mode in ("train", "val"):
            assert gt_df is not None and ids is not None
            self.samples = []
            for idx in ids:
                fname = f"{idx:04d}.jpg"
                row = gt_df[gt_df["data"] == idx]
                if len(row) == 0:
                    continue
                x = float(row.iloc[0]["Fovea_X"])
                y = float(row.iloc[0]["Fovea_Y"])
                self.samples.append((fname, x, y))
        else:  # test
            assert ids is not None
            self.samples = [(f"{idx:04d}.jpg", None, None) for idx in ids]

    def __len__(self):
        return len(self.samples)

    def _load_image(self, fname):
        """Load and decode image from disk (with class-level cache)."""
        if fname in self._image_cache:
            # Return a copy to avoid mutating the cached array
            return self._image_cache[fname].copy()

        img_path = str(self.img_dir / fname)
        img_bytes = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Cache a copy to keep the cache pristine
        self._image_cache[fname] = image.copy()
        return image

    def _load_preprocessed(self, fname):
        """Load and fully preprocess an image once, cache the result.

        Returns (preprocessed_image, inter_scale, pre_h, pre_w).
        Subsequent calls return cached copy, avoiding repeated
        JPEG decode + FOV + Illumination + CLAHE every epoch.
        """
        if fname in self._preproc_cache:
            img, inter_scale, pre_h, pre_w = self._preproc_cache[fname]
            return img.copy(), inter_scale, pre_h, pre_w

        # --- Run the full pipeline once ---
        image = self._load_image(fname)
        orig_h, orig_w = image.shape[:2]

        # Intermediate resize
        inter_scale = 1.0
        if USE_PREPROCESSING and PREPROCESS_INTER_SIZE:
            max_dim = max(orig_h, orig_w)
            if max_dim > PREPROCESS_INTER_SIZE:
                inter_scale = PREPROCESS_INTER_SIZE / max_dim
                new_w = round(orig_w * inter_scale)
                new_h = round(orig_h * inter_scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pre_h, pre_w = image.shape[:2]

        # Preprocessing pipeline (FOV → Illumination → CLAHE)
        image = _fundus_preprocess(image)

        # Cache
        self._preproc_cache[fname] = (image.copy(), inter_scale, pre_h, pre_w)
        return image, inter_scale, pre_h, pre_w

    def _pixel_normalize(self, image):
        """Apply ImageNet normalization and convert to CHW tensor."""
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        return image.transpose(2, 0, 1)  # HWC → CHW

    def __getitem__(self, idx):
        fname, x_gt, y_gt = self.samples[idx]

        # ================================================================
        #   Load & preprocess (cached after first call)
        #   Returns: preprocessed image, inter_scale, pre_h, pre_w
        # ================================================================
        image, inter_scale, pre_h, pre_w = self._load_preprocessed(fname)

        if self.mode in ("train", "val"):
            # Convert GT from original pixel coords → pre-resize normalized coords
            # x_gt * inter_scale = pixel position in pre-resize image
            # divide by pre_w → normalized in pre-resize space
            coord = np.array([
                x_gt * inter_scale / pre_w,
                y_gt * inter_scale / pre_h,
            ], dtype=np.float32)

            # Aspect-ratio preserving resize + pad (with coordinate transform)
            image, (pad_x, pad_y, scale) = _aspect_preserve_resize_pad(image)
            coord[0] = (coord[0] * pre_w * scale + pad_x) / IMG_SIZE
            coord[1] = (coord[1] * pre_h * scale + pad_y) / IMG_SIZE

            # Data augmentation
            if self.augment:
                image, coord = apply_geometric_augmentation(image, coord)

            # Pixel normalization
            image = self._pixel_normalize(image)

            return (
                torch.from_numpy(image),
                torch.from_numpy(coord),
            )

        else:
            # Test mode: apply transforms and return params for denormalization
            image, (pad_x, pad_y, scale) = _aspect_preserve_resize_pad(image)
            image = self._pixel_normalize(image)

            # Return pre_h, pre_w (intermediate size) + inter_scale for origin denormalization
            return (
                torch.from_numpy(image),
                fname,
                pre_h,
                pre_w,
                pad_x,
                pad_y,
                scale,       # IMG_SIZE / max(pre_h, pre_w)
                inter_scale, # 1.0 or PREPROCESS_INTER_SIZE / max(orig_h, orig_w)
            )


# ========================================================================
#  DataLoader factories
# ========================================================================

def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Get train and val dataloaders (70/30 split from config)."""
    gt_df = pd.read_excel(GT_FILE)

    train_dataset = FundusDataset(
        img_dir=TRAIN_IMG_DIR,
        gt_df=gt_df,
        ids=TRAIN_IDS,
        mode="train",
        augment=USE_AUGMENTATION,
    )
    val_dataset = FundusDataset(
        img_dir=TRAIN_IMG_DIR,
        gt_df=gt_df,
        ids=VAL_IDS,
        mode="val",
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def get_test_dataloader(batch_size=1, num_workers=0):
    """Get test dataloader (batch_size=1 for per-image prediction)."""
    test_dataset = FundusDataset(
        img_dir=TEST_IMG_DIR,
        gt_df=None,
        ids=TEST_IDS,
        mode="test",
        augment=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_loader


# ========================================================================
#  Cropped Fundus Dataset (for cascade fine model)
# ========================================================================

class CroppedFundusDataset(Dataset):
    """随机裁剪 ROI 训练数据集。

    每次从原图中以 GT 为中心随机偏移（jitter）裁剪固定尺寸区域，
    让黄斑在输入图中的位置随机，迫使模型学会真正的空间定位，
    消除中心偏差。

    Parameter:
        crop_size: 裁剪尺寸（原图像素空间）
        jitter_std: GT 偏移标准差（px），匹配粗模型误差分布
    """

    def __init__(self, img_dir, gt_df, ids, crop_size=CROP_TRAIN_SIZE,
                 jitter_std=CROP_JITTER_STD, augment=False):
        self.img_dir = Path(img_dir)
        self.gt_df = gt_df
        self.ids = ids
        self.crop_size = crop_size
        self.jitter_std = jitter_std
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        # Load full image
        img_path = self.img_dir / f"{sid:04d}.jpg"
        img_bytes = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # GT in original pixels
        row = self.gt_df[self.gt_df["data"] == sid].iloc[0]
        gt_x = float(row["Fovea_X"])
        gt_y = float(row["Fovea_Y"])
        half = self.crop_size // 2

        # Jittered crop center (Gaussian, clipped to stay within image)
        jx = int(np.clip(gt_x + np.random.randn() * self.jitter_std,
                         half, w - half))
        jy = int(np.clip(gt_y + np.random.randn() * self.jitter_std,
                         half, h - half))

        # Crop
        x1, y1 = jx - half, jy - half
        crop = image[y1:y1 + self.crop_size, x1:x1 + self.crop_size].copy()

        # Full preprocessing pipeline
        crop = _fundus_preprocess(crop)

        # Aspect-preserve resize + pad (no-op for square crops)
        crop, (pad_x, pad_y, scale) = _aspect_preserve_resize_pad(crop)

        # GT in crop coords (normalized to [0,1])
        gt_in_crop = np.array([
            (gt_x - x1) / self.crop_size,
            (gt_y - y1) / self.crop_size,
        ], dtype=np.float32)

        # Adjust for padding
        gt_in_crop[0] = (gt_in_crop[0] * self.crop_size * scale + pad_x) / IMG_SIZE
        gt_in_crop[1] = (gt_in_crop[1] * self.crop_size * scale + pad_y) / IMG_SIZE

        # Augmentation
        if self.augment:
            crop, gt_in_crop = apply_geometric_augmentation(crop, gt_in_crop)

        # Pixel normalization
        crop = crop.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crop = (crop - mean) / std
        crop = crop.transpose(2, 0, 1)

        return torch.from_numpy(crop), torch.from_numpy(gt_in_crop)


# ========================================================================
#  Heatmap Dataset
# ========================================================================

def generate_gaussian_heatmap(coord_norm, heatmap_size=HEATMAP_SIZE, sigma=HEATMAP_SIGMA):
    """生成二维高斯热图作为位置监督。

    Args:
        coord_norm: (x, y) 归一化坐标，取值 [0, 1]
        heatmap_size: 热图分辨率
        sigma: 高斯核标准差（热图像素空间）

    Returns:
        heatmap: [heatmap_size, heatmap_size] float32, 峰值 1.0
    """
    # 坐标映射到热图空间
    cx = coord_norm[0] * (heatmap_size - 1)
    cy = coord_norm[1] * (heatmap_size - 1)

    y_grid, x_grid = np.meshgrid(
        np.arange(heatmap_size, dtype=np.float32),
        np.arange(heatmap_size, dtype=np.float32),
        indexing='ij',
    )
    dist2 = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    heatmap = np.exp(-dist2 / (2 * sigma ** 2))
    # 确保峰值 = 1.0
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap.astype(np.float32)


class HeatmapFundusDataset(Dataset):
    """热图监督数据集：返回 (image, heatmap, coords).

    与 FundusDataset 共用预处理流水线，但额外生成 GT 高斯热图。
    """

    _preproc_cache = {}

    @classmethod
    def clear_cache(cls):
        cls._preproc_cache.clear()

    def __init__(self, img_dir, gt_df, ids, mode="train", augment=False):
        self.img_dir = Path(img_dir)
        self.mode = mode
        self.augment = augment and (mode == "train")

        self.samples = []
        for idx in ids:
            fname = f"{idx:04d}.jpg"
            row = gt_df[gt_df["data"] == idx]
            if len(row) == 0:
                continue
            x = float(row.iloc[0]["Fovea_X"])
            y = float(row.iloc[0]["Fovea_Y"])
            self.samples.append((fname, x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, x_gt, y_gt = self.samples[idx]

        # Cached preprocessed load (avoid JPEG decode + FOV/CLAHE every epoch)
        if fname in self._preproc_cache:
            image, inter_scale, pre_h, pre_w = self._preproc_cache[fname]
            image = image.copy()
        else:
            img_path = self.img_dir / fname
            img_bytes = np.fromfile(img_path, dtype=np.uint8)
            image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]

            # Intermediate resize
            inter_scale = 1.0
            if USE_PREPROCESSING and PREPROCESS_INTER_SIZE:
                max_dim = max(orig_h, orig_w)
                if max_dim > PREPROCESS_INTER_SIZE:
                    inter_scale = PREPROCESS_INTER_SIZE / max_dim
                    new_w = round(orig_w * inter_scale)
                    new_h = round(orig_h * inter_scale)
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            pre_h, pre_w = image.shape[:2]

            # Preprocessing pipeline
            image = _fundus_preprocess(image)

            # Cache
            self._preproc_cache[fname] = (image.copy(), inter_scale, pre_h, pre_w)

        # Convert GT to normalized coords (in pre-resize space)
        coord = np.array([
            x_gt * inter_scale / pre_w,
            y_gt * inter_scale / pre_h,
        ], dtype=np.float32)

        # Aspect-preserve resize + pad
        image, (pad_x, pad_y, scale) = _aspect_preserve_resize_pad(image)
        coord[0] = (coord[0] * pre_w * scale + pad_x) / IMG_SIZE
        coord[1] = (coord[1] * pre_h * scale + pad_y) / IMG_SIZE

        # Augmentation
        if self.augment:
            image, coord = apply_geometric_augmentation(image, coord)

        # Generate GT heatmap
        heatmap = generate_gaussian_heatmap(coord)  # [H_hm, W_hm]

        # Pixel normalization
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)

        return (
            torch.from_numpy(image),        # [3, 512, 512]
            torch.from_numpy(heatmap),      # [64, 64]
            torch.from_numpy(coord),        # [2]
        )


# ========================================================================
#  Mixup
# ========================================================================

def mixup_batch(images, coords, alpha=MIXUP_ALPHA):
    """Apply Mixup augmentation to a batch.

    Args:
        images: [B, 3, H, W] tensor
        coords: [B, 2] tensor (normalized)
        alpha: Beta distribution parameter

    Returns:
        mixed_images, mixed_coords, lam
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_coords = lam * coords + (1 - lam) * coords[index]

    return mixed_images, mixed_coords, lam
