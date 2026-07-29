"""Configuration for Fovea Localization project."""

import os
from pathlib import Path

# ========== Paths ==========
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "数据集"
TRAIN_IMG_DIR = DATA_DIR / "training" / "fundus color images"
TEST_IMG_DIR = DATA_DIR / "testing" / "fundus color images"
GT_FILE = DATA_DIR / "training" / "fovea_localization_training_GT.xlsx"

OUTPUT_DIR = BASE_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LOG_DIR = OUTPUT_DIR / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ========== Data split ==========
# Total 100 labeled images: 70 train, 30 val
TRAIN_IDS = list(range(1, 71))       # 0001 ~ 0070
VAL_IDS = list(range(71, 101))       # 0071 ~ 0100
TEST_IDS = list(range(101, 201))     # 0101 ~ 0200 (no GT)

# ========== Image Preprocessing ==========
IMG_SIZE = 512               # 训练输入尺寸

# ========== Model ==========
BACKBONE = "resnet34"
PRETRAINED = True
NUM_CLASSES = 1000           # 预训练分类数，之后替换分类头

# ========== Training ==========
BATCH_SIZE = 8               # 根据 RTX 4060 6GB 显存调整
EPOCHS = 150
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# ========== Loss weights ==========
ALPHA_MSE_ED = 0.5           # 坐标损失中 MSE vs ED 比例

# ========== Augmentation ==========
USE_AUGMENTATION = True
ROTATION_DEGREES = 15
BRIGHTNESS_LIMIT = 0.2
CONTRAST_LIMIT = 0.2

# ========== TTA ==========
USE_TTA = True               # 测试时翻转增强

# ========== Logging ==========
LOG_INTERVAL = 10            # 每 N 个 batch 打印一次
EVAL_INTERVAL = 1            # 每 N 个 epoch 验证一次
SAVE_TOP_K = 3               # 保存最优的 K 个模型
EARLY_STOP_PATIENCE = 15     # 验证 ED 连续 N 轮不下降则早停

# ========== Mixup ==========
MIXUP_ALPHA = 0.2            # Beta 分布参数，0=关闭，0.2=轻度混合
MIXUP_PROB = 0.5             # 每个 batch 应用 Mixup 的概率

# ========== SE Block ==========
USE_SE = True                 # 使用 SE 通道注意力

# ========== Preprocessing Pipeline ==========
USE_PREPROCESSING = True      # 启用完整预处理流水线（FOV + 光照归一化 + CLAHE）
PREPROCESS_INTER_SIZE = 1024  # 预处理前先缩放到该尺寸（最大边），避免在 3000px 级别原图上运行

# --- FOV 检测 ---
USE_FOV_MASK = True           # FOV 检测 + 掩膜（消除黑色背景干扰）

# --- 光照归一化 ---
USE_ILLUM_CORRECTION = True   # 光照归一化（大核高斯模糊消除渐晕）
ILLUM_KERNEL_RATIO = 4        # 高斯核大小 = min(H,W) / ILLUM_KERNEL_RATIO，取奇数

# --- CLAHE ---
USE_CLAHE = True              # CLAHE 对比度增强
CLAHE_CLIP_LIMIT = 3.0        # CLAHE clip limit（上调至 3.0 增加增强强度）
CLAHE_TILE_SIZE = 8           # CLAHE tile grid size (8×8)

# --- 保持纵横比缩放 + 填充 ---
USE_ASPECT_PRESERVE = True    # 保持纵横比缩放 + 填充至正方形（消除拉伸变形）

# ========== Multi-scale training ==========
USE_MULTI_SCALE = False       # 多尺度训练（最终关闭，收益有限且降低训练速度）
MULTI_SCALE_SIZES = [384, 448, 512, 576, 640]  # 候选尺寸

# ========== Coarse Localization (Two-stage cascade) ==========
COARSE_IMG_SIZE = 256          # 粗模型输入尺寸
COARSE_BACKBONE = "resnet18"   # 粗模型主干
COARSE_BATCH_SIZE = 32         # 粗模型 batch（256×256 很轻量）
COARSE_EPOCHS = 50             # 粗模型 epoch
COARSE_LR = 1e-3               # 粗模型学习率（可用更高）
COARSE_WEIGHT_DECAY = 1e-4
COARSE_EARLY_STOP = 10         # 粗模型早停
COARSE_CROP_SIZE = 512         # 在原图空间裁剪的 ROI 大小（像素）
COARSE_SAVE_TOP_K = 3

# ========== Cropped Fine Model Training (cascade) ==========
CROP_TRAIN_SIZE = 512          # 裁剪训练输入尺寸
CROP_JITTER_STD = 60           # GT jitter std（匹配粗模型误差 ~57px）

# ========== Heatmap Model ==========
HEATMAP_SIZE = 64              # 热图输出分辨率 (64×64)
HEATMAP_SIGMA = 2.0            # GT 热图高斯标准差（热图像素空间）
HEATMAP_TEMPERATURE = 0.1      # Soft-Argmax 温度参数（越小越尖锐）
HEATMAP_LOSS_ALPHA = 0.5       # 热图损失 vs 坐标损失的比例
HEATMAP_LR = 2e-4              # 热图模型学习率（可略高于坐标回归）
BATCH_SIZE_HEATMAP = 8         # 热图模型 batch size
EPOCHS_HEATMAP = 100           # 热图模型训练轮数
