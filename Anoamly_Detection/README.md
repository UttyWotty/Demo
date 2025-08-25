# Cycle Time Anomaly Detection Demo

## 🎯 Project Overview

This project implements **advanced machine learning algorithms** for detecting anomalies in cycle time data from eMoldino IoT sensors. The system analyzes cycle time and temperature data to identify unusual patterns that could indicate process issues, equipment degradation, or quality problems.

## 🧠 Machine Learning Approach

### Anomaly Detection Algorithms
- **Isolation Forest**: Unsupervised outlier detection
- **Local Outlier Factor (LOF)**: Density-based anomaly detection
- **Statistical Methods**: Z-score and IQR-based detection
- **Time-Series Anomalies**: Seasonal decomposition and trend analysis
- **Ensemble Methods**: Combining multiple algorithms for robust detection

### Key Features
- **Multi-Algorithm Approach**: Combines statistical and ML methods
- **Real-time Processing**: Stream processing for live anomaly detection
- **Feature Engineering**: Advanced sensor data preprocessing
- **Adaptive Thresholds**: Dynamic threshold adjustment based on historical data
- **Confidence Scoring**: Probabilistic anomaly scoring

## 🔧 Technical Implementation

### Technologies Used
- **Machine Learning**: scikit-learn, scipy
- **Data Processing**: pandas, numpy
- **Time-Series Analysis**: statsmodels, seasonal decomposition
- **Visualization**: matplotlib, seaborn, plotly
- **API Framework**: FastAPI, uvicorn
- **Containerization**: Docker
- **Configuration**: YAML-based parameter management

### Anomaly Detection Pipeline
1. **Data Ingestion**: Real-time sensor data collection
2. **Preprocessing**: Noise filtering, outlier removal, normalization
3. **Feature Extraction**: Statistical features, rolling windows, gradients
4. **Multi-Model Detection**: Parallel execution of different algorithms
5. **Ensemble Scoring**: Weighted combination of detection results
6. **Alert Generation**: Configurable alerting and notification system

## 📊 Detection Capabilities

### Anomaly Types Detected
- **Equipment Degradation**: Gradual performance decline
- **Process Deviations**: Sudden parameter changes
- **Sensor Malfunctions**: Data quality issues
- **Seasonal Anomalies**: Unexpected seasonal patterns
- **Contextual Anomalies**: Situation-dependent outliers

### Performance Metrics
- **Precision**: Minimizing false positives
- **Recall**: Detecting true anomalies
- **F1-Score**: Balanced performance measure
- **Detection Latency**: Real-time response time
- **Confidence Intervals**: Uncertainty quantification

## 🚀 API & Deployment

### RESTful API Endpoints
```bash
# Health check
GET /health

# Real-time anomaly detection
POST /detect/realtime
{
  "cycle_time": 45.2,
  "temperature": 240.5,
  "timestamp": "2024-01-01T10:00:00Z"
}

# Batch anomaly detection
POST /detect/batch
{
  "data": [...],
  "algorithm": "isolation_forest"
}

# Model configuration
GET /config
PUT /config
```

### Docker Deployment
```bash
# Build container
docker build --platform linux/amd64 -t anomaly-detection:latest .

# Run container
docker run -d -p 8000:8000 --name anomaly-detector anomaly-detection:latest
```

## 📁 Project Structure

```
anomaly_detection_demo/
├── analyzer.py              # Main analysis interface
├── anomaly_detectors/       # ML algorithm implementations
│   ├── isolation_forest.py
│   ├── lof_detector.py
│   ├── statistical.py
│   └── ensemble.py
├── api/                     # FastAPI application
│   └── app.py
├── configs/                 # Configuration files
│   └── anomaly_detection.yaml
├── utils/                   # Utility functions
├── requirements.txt         # Dependencies
├── Dockerfile              # Container configuration
└── data/                   # Sample datasets and results
```

## 💡 Business Impact

- **Predictive Maintenance**: Early warning system for equipment issues
- **Quality Assurance**: Automatic detection of quality deviations
- **Process Optimization**: Data-driven process parameter adjustments
- **Cost Reduction**: Minimized downtime and waste
- **Operational Excellence**: Continuous monitoring and improvement

## 🔬 Advanced Features

### Configurable Detection
- **Sensitivity Tuning**: Adjustable detection sensitivity
- **Multi-Variate Analysis**: Simultaneous analysis of multiple sensors
- **Temporal Windows**: Configurable time window analysis
- **Baseline Learning**: Adaptive baseline establishment

### Integration Capabilities
- **IoT Platform Integration**: Direct sensor data ingestion
- **Alert Systems**: Email, SMS, and webhook notifications
- **Dashboard Integration**: Real-time visualization
- **Data Export**: CSV, JSON, and database export

## 🎯 Use Cases

- **Manufacturing Lines**: Continuous monitoring of injection molding processes
- **Equipment Health**: Predictive maintenance scheduling
- **Quality Control**: Automatic quality deviation detection
- **Process Control**: Real-time process parameter monitoring

---

*This project demonstrates expertise in unsupervised machine learning, real-time data processing, IoT analytics, and industrial process optimization.*

