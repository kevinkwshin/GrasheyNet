# GrasheyNet
Cascaded Landmark Detection Model for Grashey X-ray Analysis

This repository contains a comprehensive implementation of a cascaded deep learning approach for automatic landmark detection in flatfeet X-ray images. The system combines RetinaNet for ROI detection and U-Net for precise landmark localization.

## 🏗️ Architecture Overview

The FlatNet system employs a two-stage cascaded approach:

1. **Stage 1 - ROI Detection**: RetinaNet identifies regions of interest in the X-ray images
2. **Stage 2 - Landmark Detection**: U-Net performs precise localization of 14 anatomical landmarks

## 📁 Repository Structure

```
FlatNet-main/
├── train.ipynb                 # Combined training notebook for both models
├── inference.ipynb             # Comprehensive inference and evaluation notebook
├── Preprocessing_VGGannotator.ipynb  # Data annotation preprocessing
├── pytorch-retinanet/          # RetinaNet implementation
│   ├── retinanet/              # Core RetinaNet modules
│   ├── train.py                # Training script
│   └── Retinanet_Training.ipynb
├── Unet/                       # U-Net implementation
│   ├── Unet.py                 # Core U-Net architecture
│   ├── trainer.py              # Training utilities
│   ├── loss.py                 # Loss functions (Dice, etc.)
│   ├── datagenerater.py        # Data loading and augmentation
│   └── Unet_train*.ipynb       # Training notebooks
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install required dependencies
pip install torch torchvision
pip install opencv-python matplotlib numpy
pip install segmentation-models-pytorch  # For advanced U-Net architectures
pip install scikit-learn pandas tqdm natsort
```

### 2. Data Preparation

Use the VGG annotator for creating annotations:
```bash
jupyter notebook Preprocessing_VGGannotator.ipynb
```

### 3. Training

#### Option A: Use the Combined Training Notebook (Recommended)
```bash
jupyter notebook train.ipynb
```

#### Option B: Train Models Separately
```bash
# Train RetinaNet
python pytorch-retinanet/train.py --dataset csv --csv_train path/to/train.csv --csv_classes path/to/classes.csv

# Train U-Net
jupyter notebook Unet/Unet_train.ipynb
```

### 4. Inference and Evaluation
```bash
jupyter notebook inference.ipynb
```

## 📊 Model Details

### RetinaNet (Stage 1 - ROI Detection)
- **Backbone**: ResNet (18/34/50/101/152)
- **Purpose**: Detect regions of interest in X-ray images
- **Output**: Bounding boxes with confidence scores
- **Metrics**: mAP, Precision, Recall, F1-Score

**Configuration:**
- Learning Rate: 1e-5
- Batch Size: 4
- Epochs: 100
- Input Size: 512×512

### U-Net (Stage 2 - Landmark Detection)
- **Architecture**: U-Net with multiple encoder options
- **Purpose**: Precise localization of 14 anatomical landmarks
- **Output**: 14 landmark coordinate predictions
- **Metrics**: Accuracy, Detection Rate, Mean Distance Error

**Configuration:**
- Learning Rate: 5e-4
- Batch Size: 4
- Epochs: 50
- Input Size: 512×512
- Landmarks: 14 anatomical points

#### Training Approaches:
1. **Separate Models**: 14 individual U-Net models (one per landmark)
2. **Multi-class Model**: Single U-Net with 14 output channels

## 📈 Features

### Training Capabilities
- **Comprehensive Training Pipeline**: Combined notebook for both models
- **Data Augmentation**: Extensive augmentation for robust training
- **Model Checkpointing**: Automatic saving of best models
- **Progress Tracking**: Real-time training monitoring
- **Multi-GPU Support**: Automatic GPU detection and utilization

### Inference and Evaluation
- **Batch Inference**: Efficient processing of multiple images
- **Performance Metrics**: Comprehensive evaluation for both models
- **Visualization**: Side-by-side comparison of predictions vs ground truth
- **Export Capabilities**: CSV export of results for analysis
- **Timing Analysis**: FPS and inference time measurements

### Evaluation Metrics

#### RetinaNet Metrics
- **mAP (mean Average Precision)**: Overall detection performance
- **Precision**: Ratio of true positive detections
- **Recall**: Coverage of ground truth objects
- **F1-Score**: Harmonic mean of precision and recall

#### U-Net Metrics
- **Accuracy**: Percentage of landmarks within distance threshold
- **Detection Rate**: Successfully detected landmarks
- **Mean Distance Error**: Average pixel distance from ground truth
- **Per-Landmark Analysis**: Individual landmark performance

## 🎯 Usage Examples

### Training with Custom Dataset

1. **Prepare your data structure:**
```
dataset/
├── images/           # X-ray images (.png)
├── labels/           # Landmark annotations (.npy)
├── annotations_train.csv  # RetinaNet training annotations
├── annotations_val.csv    # RetinaNet validation annotations
└── classes.csv       # Class definitions
```

2. **Update configuration in train.ipynb:**
```python
# RetinaNet Configuration
RETINANET_DATA_PATHS = {
    'csv_train': './dataset/annotations_train.csv',
    'csv_val': './dataset/annotations_val.csv',
    'csv_classes': './dataset/classes.csv'
}

# U-Net Configuration
UNET_DATA_PATHS = {
    'image_path': './dataset/images/',
    'label_path': './dataset/labels/',
}
```

3. **Run training:**
```python
# In train.ipynb, uncomment and run the training cells
# Models will be saved automatically during training
```

### Running Inference

1. **Update model paths in inference.ipynb:**
```python
RETINANET_CONFIG = {
    'MODEL_PATH': './retinanet_weights/best_model.pt',
    # ... other config
}

UNET_CONFIG = {
    'MODEL_PATH': './unet_weights/',
    # ... other config
}
```

2. **Run complete evaluation pipeline:**
```python
# Execute the complete inference pipeline
final_report = run_complete_inference_pipeline()
```

## 📋 Dataset Format

### RetinaNet Annotations (CSV format)
```csv
image_path,x1,y1,x2,y2,class_name
path/to/image1.png,100,100,200,200,foot
path/to/image2.png,150,120,250,220,foot
```

### U-Net Annotations
- **Images**: PNG format in specified directory
- **Labels**: NumPy arrays (.npy) containing landmark coordinates or segmentation masks
- **Naming**: Matching filenames between images and labels

### Classes Definition
```csv
class_name,class_id
foot,0
```

## 🔧 Advanced Configuration

### Model Architecture Options

**RetinaNet Backbones:**
- ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152

**U-Net Encoders:**
- VGG16, EfficientNet, ResNet variants
- Attention mechanisms: SCSE, Channel Attention

### Hyperparameter Tuning
- Learning rates, batch sizes, epochs configurable
- Distance thresholds for landmark accuracy
- Confidence thresholds for detection
- Data augmentation parameters

## 📊 Performance Benchmarks

### Typical Results (Example)
- **RetinaNet mAP**: 0.85-0.95 (depending on dataset)
- **U-Net Accuracy**: 90-95% (within 10-pixel threshold)
- **Inference Speed**: 15-30 FPS (depending on hardware)

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in configuration
2. **Model Loading Errors**: Check file paths and model compatibility
3. **Data Loading Issues**: Verify dataset structure and file formats

### Performance Optimization
- Use mixed precision training for faster training
- Adjust batch sizes based on available GPU memory
- Consider model pruning for deployment

## 📚 References

- **RetinaNet**: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
- **U-Net**: "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎓 Citation

If you use this work in your research, please cite:


## 📞 Support

For questions and support, please open an issue in the GitHub repository.
