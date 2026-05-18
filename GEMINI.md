# Neo Jaundice Project

## Project Overview
Neo Jaundice is a Machine Learning and Deep Learning project focused on detecting neonatal jaundice from images and predicting bilirubin levels. It combines both classical machine learning and deep learning approaches to analyze images and metadata.

The project features:
- **Deep Learning Models:** Built with PyTorch, utilizing architectures like ResNet, Vision Transformers (ViT), and Swin Transformers to process image data directly.
- **Enhanced Classical ML Models:** A suite of enhanced classical algorithms (Logistic Regression, Random Forest, XGBoost, and MLP) built using `scikit-learn` and `xgboost`. These models use automatically optimized thresholds (Youden Index), stratified cross-validation, and class imbalance handling for classification tasks.
- **Data Preprocessing & Feature Extraction:** Custom pipelines (using `opencv-python`, `pandas`, and `numpy`) to prepare image data and tabular data.

## Key Files and Directories
- `train.py`: Main PyTorch deep learning training script. Handles dataset loading, model training loop, and evaluation for Deep Learning architectures.
- `ENHANCED_MODELS_README.md`: Documentation for the enhanced classical machine learning pipeline.
- `run_all_enhanced_models.py`: Batch script to sequentially run all enhanced classical ML models and generate a comprehensive comparison.
- `compare_enhanced_models.py`: Utility to generate comparison tables for previously executed enhanced ML models.
- `requirements.txt`: Defines the Python environment dependencies (including `torch`, `scikit-learn`, `xgboost`, `opencv-python`).
- `data_preprocessing.py` / `feature_extraction.py`: Scripts used to process the image dataset and extract features before training.
- `results/`: Directory where performance metrics, confusion matrices, and comparison reports are saved.
- `models/`: Directory where trained model checkpoints (`.pkl`, `.pth`) are saved.

## Building and Running

### Running Deep Learning Models
The deep learning models can be trained by executing the main training script. Ensure paths in `train.py` are properly configured to your dataset before running.
```bash
python train.py
```

### Running Enhanced Classical Models
You can run the full suite of classical machine learning models sequentially using the batch script:
```bash
python run_all_enhanced_models.py
```
Alternatively, models can be run individually (e.g., `python XGBoost_Enhanced.py`).
To just regenerate the comparison report from existing results, run:
```bash
python compare_enhanced_models.py
```

## Development Conventions
- **Reproducibility:** Random seeds are set to 42 across enhanced models to ensure reproducible results.
- **Handling Class Imbalance:** The project uses Stratified K-Fold cross validation and applies class weights (e.g., `class_weight='balanced'`, `scale_pos_weight`) to manage dataset imbalances.
- **Evaluation:** The models emphasize comprehensive metrics. Classification is evaluated on Accuracy, Sensitivity, Specificity, F1-Score, and AUC. Threshold optimization is prioritized using the Youden Index.
- **Resource Usage:** Models are set to use all available CPU cores (`n_jobs=-1`) during training and CV for the classical models. Deep learning scripts attempt to use CUDA if available (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`).
