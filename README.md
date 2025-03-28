# Cancer-kinase-activity-prediction
Multi-Task Graph Neural Network for predicting kinase inhibitor activity using molecular graph representations. Leverages advanced deep learning techniques to classify compound interactions across multiple kinase targets. Utilizes ChEMBL data, ensemble learning, and GNNs to improve drug discovery computational screening processes.
# Multi-Task Graph Neural Network for Kinase Inhibitor Prediction

This repository contains code for a Graph Neural Network (GNN) model that predicts the activity of chemical compounds against multiple kinase targets. The model uses molecular structures represented as graphs to identify potential inhibitors across different kinase families.

## Dataset

The original data was sourced from Kaggle and enriched with additional information: (https://www.kaggle.com/datasets/xiaotawkaggle/inhibitors/)
- ChEMBL IDs were used to retrieve SMILES representations and IC50 values from the ChEMBL database
- The dataset includes compounds tested against multiple kinase targets
- Activity is encoded as a binary classification problem (active/inactive)

## Features

This repository contains two implementations of the same concept:

### Basic Multi-Task GNN Model
- Simple GCN architecture with two graph convolutional layers
- Multi-task output layer for all kinase targets
- Binary cross-entropy loss function
- Performance: AUC: 0.7775, Accuracy: 0.7439, F1 Score: 0.8186

### Enhanced Multi-Task GNN Ensemble
- Three-layer GCN architecture with additional regularization techniques:
  - Dropout for preventing overfitting
  - Batch normalization for stable training
  - Gradient clipping to prevent exploding gradients
- Early stopping implementation to prevent overfitting
- Learning rate scheduling to improve convergence
- Ensemble of multiple models for robust predictions
- Performance: Accuracy: 0.8147, F1 Score: 0.8660

## Improvements in the Enhanced Model

The enhanced model outperforms the basic implementation due to several key improvements:

1. **Ensemble Learning**: By combining predictions from multiple models with different architectures and hyperparameters, the ensemble approach reduces variance and improves overall robustness.

2. **Regularization Techniques**: The enhanced model incorporates dropout, batch normalization, and weight decay to prevent overfitting and improve generalization.

3. **Deeper Architecture**: The additional convolutional layer allows the model to learn more complex molecular patterns relevant to kinase inhibition.

4. **Learning Rate Scheduling**: The ReduceLROnPlateau scheduler adjusts the learning rate based on validation performance, leading to better convergence.

5. **Early Stopping**: This prevents overfitting by stopping training when validation performance stops improving, retaining the best model weights.

6. **Hyperparameter Diversity**: The ensemble uses different hyperparameters for each model, increasing the diversity of learned features.

## Usage

The code processes molecular SMILES strings into graph representations using PyTorch Geometric's utilities. The model can be trained on balanced datasets and evaluated using standard metrics like AUC, accuracy, and F1 score.

## Requirements

- PyTorch
- PyTorch Geometric
- RDKit
- scikit-learn
- pandas
- numpy
