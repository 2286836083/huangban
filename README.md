# 黄斑中心凹定位 — Fovea Localization

> 基于眼底图像的黄斑中心凹坐标回归项目  
> Fovea centralis coordinate regression from fundus images  

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](数据集/License-GAMMA_1019.pdf)

---

## 项目简介 | Overview

本项目基于 **百度飞桨 AI Studio 黄斑中心凹定位赛题**（[比赛链接](https://aistudio.baidu.com/competition/detail/806/0/introduction)）提供的眼底影像数据，使用深度学习模型对 **黄斑中心凹（Fovea）** 进行精确定位。项目从官方基线（ResNet50 + 3FC）出发，系统性地探索了多种改进方案：

| 实验 | 方案 | Best AED | 像素误差 |
|:---:|:---|:--------:|:--------:|
| Exp-00 | 官方基线 (ResNet50 + 3FC) | 0.0204 | 5.2 px |
| Exp-01 | ResNet34 坐标回归 | 0.0158 | — |
| Exp-02 | ResNet34 5 折 CV | **0.0138** | — |
| Exp-03 | + 预处理流水线 (ResNet34) | 0.0149 | — |
| Exp-05 | 热图模型 (ResNet34) | 0.0155 | — |
| Exp-06 | U-Net 热图 (ResNet34) | 0.0165 | — |
| Exp-07 | 5 折集成 + TTA | **0.0112** | **3.1 px** |

> **最佳方案**：预处理 + ResNet34 5 折热图模型 + TTA → **AED 0.0112，约 3.1 像素误差**

详细报告见 [`experiment_report.md`](./experiment_report.md)

---

## 项目结构 | Structure

```
新工作区/
├── src/
│   ├── config.py          # 全局配置（路径、超参数、预处理开关）
│   ├── dataset.py         # 数据集加载与增强
│   ├── model.py           # 模型定义（ResNet + 坐标头 / 热图头）
│   ├── loss.py            # 损失函数（MSE + ED 组合）
│   └── utils.py           # 工具函数（可视化、指标计算）
├── 数据集/                 # 百度飞桨赛题数据集（需自行下载）
│   ├── training/
│   └── testing/
├── train_baseline_origin.py          # Exp-00: 官方基线复现
├── train_comparison.py               # Exp-01: Backbone 对比
├── train_5fold_comparison.py         # Exp-02: 5 折 CV backbone 对比
├── train_5fold_preprocess.py         # Exp-03: 预处理流水线 5 折 CV
├── train_5fold_heatmap.py            # Exp-05: 热图模型 5 折 CV
├── train_5fold_unet_heatmap.py       # Exp-06: U-Net 热图 5 折 CV
├── train_simple_baseline.py          # 简化基线
├── ensemble_eval.py                  # Exp-07: 模型集成 + TTA 评估
├── find_best_samples.py              # 最佳/最差样本分析
├── visualize_progression.py          # 训练过程可视化
├── experiment_report.md              # 完整实验报告
├── requirements.txt                  # 依赖
├── .gitignore
└── README.md
```

---

## 环境配置 | Installation

```bash
# 推荐 Python 3.9+
pip install -r requirements.txt
```

主要依赖：
- **PyTorch** ≥ 2.0
- **TorchVision** ≥ 0.15
- **OpenCV**, **NumPy**, **Pandas**, **Matplotlib**, **scikit-learn**

---

## 使用说明 | Usage

### 准备数据
1. 从 [百度飞桨 AI Studio 赛题页](https://aistudio.baidu.com/competition/detail/806/0/introduction) 下载数据集
2. 将 `training/` 和 `testing/` 放入 `数据集/` 目录

### 训练模型

| 命令 | 说明 |
|:-----|:-----|
| `python train_baseline_origin.py` | 官方基线复现 |
| `python train_comparison.py` | Backbone 对比 (RN18/34/50) |
| `python train_5fold_comparison.py` | 5 折 CV 对比 |
| `python train_5fold_preprocess.py` | 预处理流水线实验 |
| `python train_5fold_heatmap.py` | 热图模型实验 |
| `python train_5fold_unet_heatmap.py` | U-Net 热图实验 |
| `python train_simple_baseline.py` | 简化快速基线 |

### 评估与集成

```bash
python ensemble_eval.py              # 5 折模型集成 + TTA
python find_best_samples.py          # 分析最佳/最差预测样本
python visualize_progression.py      # 可视化训练过程
```

---

## 主要方法 | Methods

### 坐标回归模型
- **Backbone**：ResNet18 / ResNet34 / ResNet50（ImageNet 预训练）
- **Head**：自适应池化 → FC 层 → 空间注意力 → 坐标输出
- **SE Block**：通道注意力增强特征判别力
- **损失**：`0.5 × MSE + 0.5 × Euclidean Distance`

### 热图模型
- 生成高斯热图（64×64）作为监督
- Soft-Argmax 解码为坐标
- 损失：`α × 热图 MSE + (1-α) × 坐标 MSE`

### 预处理流水线
1. **FOV 掩膜** — 消除黑色背景干扰
2. **光照归一化** — 大核高斯模糊去除渐晕
3. **CLAHE** — 对比度增强
4. **保纵横比缩放 + 填充** — 消除拉伸变形

### 级联架构（可选）
1. **粗模型**（256×256）快速预测大致位置
2. 根据粗预测裁剪 ROI
3. **精模型**（512×512）在 ROI 上精细预测

---

## 参考文献 | References

- [百度飞桨 AI Studio · 黄斑中心凹定位赛题](https://aistudio.baidu.com/competition/detail/806/0/introduction) — 比赛与数据来源
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- 数据集许可见 [`数据集/License-GAMMA_1019.pdf`](./数据集/License-GAMMA_1019.pdf)