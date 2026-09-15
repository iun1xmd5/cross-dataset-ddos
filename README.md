# Bridging the Generalization Gap

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)](https://www.tensorflow.org/)

Reproducibility artefacts for the paper:

> **Bridging the Generalization Gap: Cross-Dataset Validation of LSTM-Based DDoS Detection in Multi-Class Network Traffic Environments**  
> *Under review* (2026)

---

## Overview

This repository contains the full pipeline for reproducing the cross-dataset validation study comparing five DDoS detection models — LSTM, BiLSTM, CNN-LSTM, Random Forest, and XGBoost — on two benchmark datasets: **CICDDos2019** (19 classes) and **CICIDS2017** (15 classes).

### Headline finding

A model that ranks first on one benchmark does **not** necessarily rank first on another. On CICDDos2019 the LSTM leads (macro F1 = 0.8470); on CICIDS2017 the CNN-LSTM leads (macro F1 = 0.9653) while LSTM ranks lowest among the deep learning models (0.9006). The ranking shift is explained by class composition: CICDDos2019 contains multiple structurally similar reflection-based amplification attacks that temporal modelling handles better, while CICIDS2017 presents a broader mix of attack families whose feature-space boundaries are more separable.

### Scope

- **5 architectures** evaluated on **2 datasets** with **19 + 15 classes**
- **Leakage-controlled pipeline**: training-only feature selection, training-only scaling, stratified 70/15/15 split, post-split sequence construction
- **Statistical testing**: Friedman → Holm-corrected Wilcoxon → McNemar
- **Computational-cost comparison**: training time, inference latency, throughput, model size

---

## Repository Contents

| Directory | Purpose |
|---|---|
| `src/` | Core Python modules: data loading, preprocessing, feature selection, model definitions, training, evaluation, statistical tests |
| `scripts/` | Dataset download and preparation scripts, plus end-to-end shell entry points |
| `reproduce_figures/` | Standalone scripts that reproduce the four figures using the paper's reported values |
| `notebooks/` | Exploratory and analysis notebooks used during development |
| `results/` | Output directory for trained model artefacts and metric dumps (gitignored) |
| `figures/` | Output directory for generated figures |
| `docs/` | Extended documentation of the methodology, leakage-control measures, and statistical procedures |

---

## Prerequisites

- **Python** ≥ 3.10
- **TensorFlow** ≥ 2.19 (GPU strongly recommended for deep learning models)
- **XGBoost** ≥ 2.0
- **scikit-learn** ≥ 1.4
- **A CUDA-capable GPU** (optional but ~10× faster for LSTM training)
- **~15 GB disk** for the raw datasets + preprocessed artefacts

The Kaggle environment used in the paper provided **two Tesla T4 GPUs (15,360 MiB each), 4 CPU cores, 31.3 GB RAM**. Comparable results were obtained with a single T4.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/iun1xmd5/cross-dataset-ddos.git
cd cross-dataset-ddos

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
