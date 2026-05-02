# Olive ML Pipeline 🌿

Machine Learning Pipeline for Olive Grove Classification using Google Earth Engine Data

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📋 Overview

This repository contains a comprehensive machine learning pipeline for olive grove classification in Tunisia using satellite imagery data from Google Earth Engine. The pipeline implements both traditional machine learning (LightGBM) and deep learning (ResNet, U-Net) approaches for multi-stage olive grove analysis.

## 🚀 Key Features

### Core Pipeline
- **Two-stage classification**: Binary detection → Intensity level classification
- **Zone-specific models**: Trained separately for SFAX, SAHEL, MIDWEST, NORTH, and SOUTH regions
- **Fast training**: Complete pipeline trains in under 20 seconds
- **High performance**: AUC scores 0.82-1.0 across all zones

### Enhanced Analytics
- **IoU metrics**: Intersection over Union calculations for segmentation evaluation
- **Performance visualization**: Automated plots for F1 scores and IoU comparisons
- **Feature importance**: Analysis of satellite bands and vegetation indices
- **Confusion matrices**: Detailed classification performance per zone

### Deep Learning Templates
- **ResNet-style CNN**: 159K parameters for tabular feature classification
- **U-Net architecture**: 6.9M parameters for spatial segmentation
- **PyTorch Dataset**: Custom dataset class for satellite imagery
- **Training utilities**: Complete training loops with validation

## 📊 Performance Results

| Zone     | Binary F1 | Binary AUC | Intensity F1 | Mean IoU |
|----------|-----------|------------|--------------|----------|
| SFAX     | 0.547     | 0.975      | 0.687        | 0.577    |
| SAHEL    | 0.415     | 0.824      | 0.986        | 0.972    |
| MIDWEST  | 0.675     | 0.907      | 0.984        | 0.968    |
| NORTH    | 0.479     | 0.817      | 1.000        | 1.000    |
| SOUTH    | 0.000     | 1.000      | N/A          | N/A      |

## 🛠️ Installation

### Prerequisites
- Python 3.14+
- Git
- Virtual environment (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/olive-ml-pipeline.git
cd olive-ml-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install lightgbm torch torchvision scikit-learn pandas numpy matplotlib seaborn geopandas
```

## 📁 Project Structure

```
olive-ml-pipeline/
├── olive_ml_pipeline.py          # Main LightGBM pipeline
├── olive_dl_models.py            # Deep learning models (ResNet, U-Net)
├── enhanced_olive_pipeline.py    # Enhanced pipeline with IoU metrics
├── GEE_OliveTunisia/            # Google Earth Engine data
│   ├── OliveTunisia_TrainingSamples_AllZones.csv
│   └── *.tif files (satellite imagery)
├── models/                      # Trained models and metrics
│   ├── *.pkl (LightGBM models)
│   ├── metrics.json
│   └── olive_ml_metrics_visualization.png
├── plots/                       # Performance visualizations
│   ├── *_feature_importance.png
│   ├── *_confusion_matrix.png
│   └── zone_performance_summary.png
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🚀 Usage

### Quick Start
```bash
# Run the complete pipeline
python olive_ml_pipeline.py

# Run enhanced pipeline with IoU metrics
python enhanced_olive_pipeline.py

# Test deep learning models
python olive_dl_models.py
```

### Training Deep Learning Models
```python
from olive_dl_models import create_olive_resnet_model, train_resnet_model

# Create and train ResNet model
model = create_olive_resnet_model(input_dim=20, num_classes=3)
# ... prepare your data ...
trained_model = train_resnet_model(model, train_loader, val_loader)
```

## 📈 Data Sources

- **Google Earth Engine**: Satellite imagery and vegetation indices
- **MODIS FAPAR/LAI**: Photosynthetically active radiation data
- **WorldCover v100**: Land cover classification
- **Ground truth**: Olive grove parcels and intensity levels

## 🤖 Model Architecture

### LightGBM Pipeline
1. **Stage 1 (Binary)**: Olive vs Non-olive classification
2. **Stage 2 (Intensity)**: Classification into extensif/intensif/hyper_intensif

### Deep Learning Models
- **ResNet**: Residual neural network for tabular features
- **U-Net**: Encoder-decoder for spatial segmentation

## 📊 Evaluation Metrics

- **F1 Score**: Harmonic mean of precision and recall
- **IoU (Jaccard Index)**: Intersection over Union for segmentation
- **AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Earth Engine for satellite data access
- Tunisian Ministry of Agriculture for ground truth data
- LightGBM and PyTorch communities for excellent ML frameworks

## 📞 Contact

For questions or collaborations, please open an issue or contact the maintainers.

---

**🌿 Happy olive grove analysis!**