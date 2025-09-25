# Digit Recognizer: Advanced CNN with Optuna Hyperparameter Optimization

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/digit-recognizer)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-2B3B4A)](https://optuna.org)

A comprehensive solution to the classic **MNIST Digit Recognition** problem on Kaggle. This project demonstrates a full machine learning pipeline, culminating in a highly accurate Convolutional Neural Network (CNN) whose hyperparameters are automatically tuned using **Optuna**.

## 🚀 Project Overview

The goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. This project implements and compares multiple approaches:
1.  **Baseline Models:** Logistic Regression, Random Forest, and XGBoost.
2.  **Deep Learning:** A custom Convolutional Neural Network (CNN) with data augmentation.
3.  **Hyperparameter Tuning:** Automated optimization of the CNN using Optuna to achieve peak performance.

The final optimized model achieves over **99.6% accuracy** on the validation set.

## 📊 Results Summary

| Model | Validation Accuracy | Key Features |
| :--- | :--- | :--- |
| **Logistic Regression** | ~91.3% | Linear baseline |
| **Random Forest** | ~96.5% | Ensemble method (200 trees) |
| **XGBoost** | ~97.7% | Gradient boosting |
| **CNN (Optuna-Optimized)** | **~99.6%** | **Data Augmentation, Batch Normalization, Optuna Tuning** |

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.11
- **Core ML:** `scikit-learn`, `XGBoost`
- **Deep Learning:** `TensorFlow` / `Keras`
- **Hyperparameter Tuning:** `Optuna`
- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`


## 🧪 Methodology

### 1. Data Preprocessing
- Pixel values normalized to `[0, 1]` range.
- Train-Validation split (90%-10%).

### 2. Baseline Models
- Established performance benchmarks with traditional ML models.

### 3. Convolutional Neural Network (CNN)
The architecture features:
- **Convolutional Layers:** Multiple `Conv2D` layers with `3x3` kernels and ReLU activation.
- **Batch Normalization:** For stable and faster training.
- **Max-Pooling:** For dimensionality reduction.
- **Dropout:** To prevent overfitting.
- **Data Augmentation:** Real-time augmentation (rotations, shifts, zooms) to improve generalization.

### 4. Hyperparameter Optimization with Optuna
Optuna was used to automatically find the best hyperparameters over 20 trials, searching for:
- **Dropout rates**
- **Number of units** in the dense layer
- **Learning rate**
- **Batch size**



## 📈 Key Findings

- **CNN Superiority:** Deep learning significantly outperforms traditional machine learning models for image classification tasks.
- **Optuna's Value:** Automated hyperparameter tuning provided a noticeable boost in validation accuracy compared to manually set defaults.
- **Data Augmentation:** Crucial for building a robust model that generalizes well.

## 🔮 Future Improvements

- Experiment with more complex architectures (e.g., ResNet, EfficientNet).
- Use **Cross-Validation** for a more reliable score estimate.
- Implement **Test Time Augmentation (TTA)** for potentially better predictions.
- Deploy the model as an interactive web app using **Gradio** or **Streamlit**.

## 🤝 Contributing

Contributions and suggestions are welcome! Please feel free to fork the repository and submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you add one).

---

**⭐ If you found this project helpful or interesting, please give it a star! ⭐**
