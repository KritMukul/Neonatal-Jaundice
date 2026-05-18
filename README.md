# Neo Jaundice Detection 🩺👶

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Classical%20ML-F7931E.svg)](https://scikit-learn.org/)

## 📖 Project Overview
Neo Jaundice is a comprehensive Machine Learning and Deep Learning project focused on detecting neonatal jaundice from clinical images and predicting bilirubin levels. By combining state-of-the-art Deep Learning models with highly optimized classical Machine Learning pipelines, this project aims to provide an accurate, non-invasive diagnostic aid for neonatal care.

### 🌟 Key Features
- **Deep Learning Architectures:** Implements Vision Transformers (ViT), Swin Transformers, and ResNet models using PyTorch to process and classify image data directly.
- **Enhanced Classical ML Models:** A robust suite of classical algorithms (Logistic Regression, Random Forest, XGBoost, and MLP) built with `scikit-learn` and `xgboost`.
- **Advanced Optimization:** Classification tasks feature automatic threshold optimization using the Youden Index, stratified cross-validation, and rigorous handling of class imbalances.
- **End-to-End Pipeline:** Includes custom data preprocessing and feature extraction modules (`opencv-python`, `pandas`, `numpy`) to prepare both raw image data and structured metadata.

---

## 📁 Repository Structure

```text
Neo_Jaundice/
├── train.py                     # Main PyTorch DL training and evaluation script
├── data_preprocessing.py        # Image and tabular data preprocessing
├── feature_extraction.py        # Feature extraction pipelines
├── ENHANCED_MODELS_README.md    # Docs specifically for the enhanced ML pipeline
├── run_all_enhanced_models.py   # Batch execution script for all classical models
├── compare_enhanced_models.py   # Utility to generate performance comparisons
├── requirements.txt             # Python dependencies
├── results/                     # Output directory for metrics, plots, and logs
└── models/                      # Output directory for saved model weights (.pth, .pkl)
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/Neo_Jaundice.git
cd Neo_Jaundice
pip install -r requirements.txt
```

### 2. Deep Learning Models
To train the deep learning architectures (ensure paths to your dataset are properly set inside the script):

```bash
python train.py
```

### 3. Enhanced Classical ML Models
You can run the complete suite of optimized classical machine learning models sequentially, which will automatically generate a comprehensive comparison report in the `results/` folder:

```bash
python run_all_enhanced_models.py
```

If you prefer to run a single model:
```bash
python XGBoost_Enhanced.py
```

To regenerate the comparison report from previously saved results:
```bash
python compare_enhanced_models.py
```

---

## 🧪 Methodology & Conventions

- **Reproducibility:** All models and splits use a fixed random seed (`42`) to guarantee reproducible outcomes across runs.
- **Handling Class Imbalance:** The dataset is rigorously handled using `StratifiedKFold` cross-validation. Algorithm-specific class weights (e.g., `class_weight='balanced'`, `scale_pos_weight`) are applied dynamically.
- **Comprehensive Evaluation:** Models are evaluated across multiple metrics including Accuracy, Sensitivity, Specificity, F1-Score, and AUC. For classical models, the classification threshold is actively optimized to maximize the Youden Index (Sensitivity + Specificity - 1).
- **Resource Management:** Classical model scripts parallelize workloads across all available CPU cores (`n_jobs=-1`), while deep learning scripts automatically detect and utilize CUDA-enabled GPUs if available.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
