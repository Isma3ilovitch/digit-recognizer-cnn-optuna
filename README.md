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

## 📁 Repository Structure
