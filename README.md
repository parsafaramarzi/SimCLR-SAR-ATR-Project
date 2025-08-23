# SimCLR SAR ATR Project

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This is a **Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR)** project using the contrastive learning framework **SimCLR**.  
It was developed as a **university research project** based on the following paper:  
👉 [Target Recognition from SAR Images Using Deep Learning and Contrastive Pretraining](https://www.sciencedirect.com/science/article/pii/S1877050922014697)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ysjMt-EeY1dCM2WFBuefTr4kaHS6_pmC?usp=sharing)

---

## 📌 Introduction

In the field of **automatic target recognition using SAR images**, deep learning methods face significant challenges when dealing with **small sample sizes**. Although deep learning has shown great promise, it often suffers from:

- **Overfitting** on limited data  
- **Gradient explosion/instability** during training  

To address these issues, this project applies **self-supervised contrastive learning (SimCLR)** on **unlabeled SAR data**. By pretraining a model to extract robust feature representations, the learned weights can then be fine-tuned on small labeled datasets, reducing overfitting and improving recognition accuracy.

<p align="center">
  <img src="/media/TargetRecognitionFromSARImagesUsingDeepLearningExample_01.png" alt="SAR-ATR" title="SAR-ATR" width="500"/>
</p>

---

## 🔎 Related Works & Challenges

- **Deep Learning in SAR Classification:** CNN-based methods significantly improved SAR image recognition compared to traditional approaches, but require large labeled datasets.  
- **Transfer Learning & Pretraining:** Pretraining on large datasets and fine-tuning on smaller ones improves performance, but domain gaps remain.  
- **Self-Supervised Learning:** Recent methods (e.g., contrastive learning) leverage **unlabeled data** to learn robust features, improving downstream SAR ATR tasks.  
- **Small-Sample Issues:** Overfitting and poor generalization remain common when SAR datasets are small, motivating augmentation, regularization, and SSL-based methods.  

---

## 🔗 Contrastive Learning

Contrastive learning trains models to **learn representations without labels**, by distinguishing between **positive pairs** (similar data) and **negative pairs** (dissimilar data).  

Key Concepts:
- **Representation Learning:** Learn a feature space where similar points cluster and dissimilar points are distant.  
- **Positive/Negative Pairs:** Positive = augmentations of the same image; Negative = augmentations from different images.  
- **Contrastive Loss (InfoNCE/NT-Xent):** Minimizes distance between positive pairs while maximizing distance between negatives.  
- **Data Augmentation:** Essential for creating diverse views (cropping, flipping, noise, jittering, etc.).  
- **Self-Supervised Paradigm:** Labels are not needed—augmentations generate pseudo-labels.  

<p align="center">
  <img src="/media/contrastive_standard.png" alt="Contrastive Learning" title="Contrastive Learning" width="500"/>
</p>

---

## ⚡ Proposed Solution: SimCLR

**SimCLR (Simple Framework for Contrastive Learning of Visual Representations)**, developed by Google Research, is applied to SAR ATR in this project.

<p align="center">
  <img src="/media/1_GuoSK8ghNX11JUlq-j0LYw.png" alt="SimCLR" title="SimCLR" width="500"/>
</p>

### Core Components
- **Self-Supervised Learning:** Leverages unlabeled SAR data by generating labels via augmentations.  
- **Data Augmentation:** Each image is augmented twice (cropping, jittering, flipping, blurring) → forms positive pairs.  
- **Neural Network Encoder:** A ResNet-based encoder extracts deep features.  
- **Projection Head:** A small MLP maps encoded features into a space where contrastive loss is applied.  
- **NT-Xent Loss:** Ensures augmented views of the same image align, while separating views of different images.  
- **Training Dynamics:** Large batch sizes yield more negative pairs, critical for strong contrastive learning.  
- **Downstream Use:** After training, the projection head is discarded; the encoder is fine-tuned on small labeled SAR ATR datasets.  

---

## 📂 Datasets

### [MSTAR (8 Classes)](https://www.kaggle.com/datasets/atreyamajumdar/mstar-dataset-8-classes)
- High-resolution SAR images of military targets and civilian vehicles.  
- Widely used benchmark dataset for SAR ATR research.  

### [SARScope](https://www.kaggle.com/datasets/kailaspsudheer/sarscope-unveiling-the-maritime-landscape)
- Designed for **SAR ship detection & segmentation**.  
- Combines HRSID + OPEN-SSDD → **6735 images**.  
- Provides diversity and robustness for maritime ATR tasks.  

---

## 📊 Results (Planned/Expected)

- Training convergence curves *(add plots here)*  
- Feature embeddings visualization *(t-SNE/UMAP plots)*  
- Classification accuracy on small labeled SAR datasets *(add table here)*  
