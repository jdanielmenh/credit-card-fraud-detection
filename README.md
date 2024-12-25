# Credit Card Fraud Detection

This repository contains a complete pipeline for detecting credit card fraud using machine learning. The dataset for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The project is structured into several stages, including exploratory data analysis (EDA), data preprocessing, model training, and evaluation. Additionally, the repository is equipped with tools for reproducibility and scalability using MLflow and Poetry.

---

**Note:** The `mlruns/` and `extras/` directories are ignored by Git and are not included in the repository.

---

## Dataset

The dataset is publicly available on Kaggle and contains anonymized credit card transactions. It includes 31 features:
- **Time**: Seconds elapsed between each transaction and the first transaction in the dataset.
- **V1-V28**: Principal components obtained through PCA (anonymized features).
- **Amount**: Transaction amount.
- **Class**: Binary target variable indicating whether a transaction is fraudulent (`1`) or not (`0`).

---

## Objectives

- Perform exploratory data analysis to understand the dataset and identify any patterns or anomalies.
- Build and evaluate a machine learning model to predict fraudulent transactions.
- Ensure the project is reproducible using MLflow for experiment tracking and Poetry for dependency management.

---

## Tools and Technologies

- **Python**: Programming language for data analysis and machine learning.
- **Pandas & NumPy**: For data manipulation and numerical computations.
- **Matplotlib & Seaborn**: For visualizing data insights.
- **Scikit-learn**: For building and evaluating the machine learning model.
- **MLflow**: For experiment tracking, versioning, and model deployment.
- **Poetry**: For managing project dependencies and virtual environments.

---

## Steps in the Project

### 1. Data Exploration and Visualization
- Conducted in the `01_eda.ipynb` notebook and the `eda.py` script.
- Key insights:
  - The dataset is highly imbalanced (fraudulent transactions constitute only ~0.17% of all transactions).
  - Some features exhibit significant differences between fraudulent and non-fraudulent transactions.

### 2. Data Preprocessing
- Performed in the preprocessing section of the training pipeline.
- Includes:
  - Standardizing the `Amount` and `Time` features.
  - Handling class imbalance using techniques like under-sampling.

### 3. Model Training
- Carried out in the `02_training.ipynb` notebook.
- Steps:
  - Used Lightgbm.
  - Used Grid Search for hyperparameter optimization.
  - Evaluated models using metrics like Precision, Recall, F1-Score, and AUC-ROC.
- MLflow was used to log experiments, model parameters, and performance metrics.

### 4. Experiment Tracking with MLflow
- MLflow tracks all experiments, including:
  - Model parameters and configurations.
  - Performance metrics and evaluation scores.

### 5. Dependency Management with Poetry
- All dependencies are defined in `pyproject.toml` and locked in `poetry.lock`.
- To install dependencies, simply run:
  ```bash
  poetry install
    ```

