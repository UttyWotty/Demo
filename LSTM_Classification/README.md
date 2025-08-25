# LSTM Production Pattern Classification Demo

## 🎯 Project Overview

This project implements a **Long Short-Term Memory (LSTM) neural network** for classifying production patterns in injection molding operations. The system analyzes time-series sensor data to predict equipment behavior and optimize manufacturing processes.

## 🧠 Machine Learning Approach

### Model Architecture
- **LSTM Neural Network** with PyTorch implementation
- **Time-series sequence processing** with configurable window sizes
- **Multi-class classification** for production patterns
- **Class imbalance handling** with SMOTE, undersampling, and weighted loss functions

### Key Features
- **Feature Engineering**: Advanced preprocessing of sensor data (temperature, pressure, cycle time)
- **Sequence Windowing**: Sliding window approach for temporal pattern recognition
- **Imbalanced Data Handling**: Multiple strategies (SMOTE, SMOTE-ENN, class weights)
- **Model Validation**: Train/validation/test split with stratified sampling
- **Performance Metrics**: Comprehensive evaluation with precision, recall, F1-score

## 🔧 Technical Implementation

### Technologies Used
- **Deep Learning**: PyTorch, torchvision
- **Data Processing**: pandas, numpy, scikit-learn
- **Imbalanced Learning**: imbalanced-learn (SMOTE)
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib

### Data Pipeline
1. **Data Loading**: CSV input with time-series sensor data
2. **Preprocessing**: Feature scaling, label encoding, sequence creation
3. **Windowing**: Create temporal sequences for LSTM input
4. **Balancing**: Apply SMOTE or undersampling for class balance
5. **Training**: LSTM model training with early stopping
6. **Inference**: Production-ready inference pipeline

## 📊 Model Performance

### Training Features
- Temperature sensors (multiple zones)
- Pressure measurements
- Cycle time variations
- Equipment parameters
- Process conditions

### Output Predictions
- Production pattern classification
- Anomaly detection capabilities
- Equipment state prediction
- Process optimization recommendations

## 🚀 Usage

```python
# Train new model
python lstm_production_pattern_pipeline.py --train --input data/production_data.csv

# Run inference
python lstm_production_pattern_pipeline.py --predict --model saved_model.pth --input new_data.csv

# Evaluate model
python lstm_production_pattern_pipeline.py --evaluate --model saved_model.pth --test_data test.csv
```

## 📁 Project Structure

```
lstm_classification_demo/
├── analyzer.py              # Main analysis interface
├── data_processing.py       # Data preprocessing utilities
├── model.py                # LSTM model definition
├── utils.py                # Training and inference utilities
├── requirements.txt        # Dependencies
├── data/                   # Sample datasets
└── models/                 # Trained model artifacts
```

## 💡 Business Impact

- **Predictive Maintenance**: Early detection of equipment issues
- **Quality Control**: Pattern recognition for defect prevention
- **Process Optimization**: Data-driven parameter tuning
- **Cost Reduction**: Minimized downtime and waste

## 🔬 Research & Development

This project demonstrates advanced applications of:
- **Time-series deep learning** in manufacturing
- **Industrial IoT data analysis** with sensor fusion
- **Production process optimization** through ML
- **Real-time pattern recognition** systems

---

*Part of comprehensive data science portfolio demonstrating expertise in deep learning, time-series analysis, and industrial IoT applications.*

