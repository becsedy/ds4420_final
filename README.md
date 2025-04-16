# Galaxy Morphology and Parameter Estimation

This repository contains code for a class project exploring two machine learning approaches in astrophysics:

1. **Galaxy Morphology Classification** using Convolutional Neural Networks (CNNs)
2. **Galaxy Mass and Redshift Estimation** using Bayesian Regression

## Project Overview

We investigate how model complexity and data augmentation impact galaxy image classification accuracy, and how a simplified Bayesian model can predict redshift and stellar mass from photometric data.

### CNN-based Classification
- Trained and evaluated three models: `SimpleCNN`, `PowerfulCNN`, and `EfficientNet_B2` (transfer learning)
- Dataset used: **Galaxy10 DECaLS**
- Training involved PyTorch with data augmentation for improved generalization

### Bayesian Regression
- Used simulated galaxy photometry (6 bands: u, g, r, i, z, y)
- Built a degree-2 polynomial regression model with a Gaussian prior to estimate redshift and stellar mass
- Computed posterior distributions to quantify prediction uncertainty

## Datasets

- **Galaxy10 DECaLS** (for image classification)  
🔗 [Download from Zenodo](https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5)

- **Galaxy Mass Prediction** (for Bayesian regression)  
  🔗 [Download from Kaggle](https://www.kaggle.com/datasets/mexwell/galaxy-mass-prediction/data)

## Authors

Yash R. Bhora  
Benjamin Ecsedy

Project for DS 4420: Machine Learning at Northeastern University.
