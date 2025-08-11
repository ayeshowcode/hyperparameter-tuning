# 🚀 Neural Network Hyperparameter Tuning

Automated hyperparameter optimization for neural networks using Keras Tuner on the diabetes dataset.

## Overview

This project demonstrates automated hyperparameter tuning by comparing a baseline neural network with an optimized version that uses Keras Tuner to find the best settings.

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook main2.ipynb
```

## Dataset
- **Source**: Pima Indian Diabetes Dataset (`diabetes.csv`)
- **Task**: Binary classification (diabetes prediction)
- **Features**: 8 medical variables (glucose, BMI, age, etc.)
- **Samples**: 768 instances

## What's Included
- 📊 Data exploration and preprocessing
- 🧪 Baseline neural network model
- 🔧 Automated hyperparameter tuning with Keras Tuner
- 📈 Performance comparison and analysis

## Hyperparameters Tuned
- **Hidden layer size**: 16-128 neurons
- **Optimizer**: Adam, SGD, RMSprop, Adadelta
- **Learning rate**: 0.0001-0.01 (log scale)

## Technologies
- TensorFlow/Keras
- Keras Tuner
- Pandas, NumPy
- Scikit-learn

## Results
The tuned model shows improved performance over the baseline, demonstrating the value of automated hyperparameter optimization.

---
*Educational project showcasing ML best practices* 🎓
