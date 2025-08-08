# Machine Learning Project: Survival Time Prediction

## Project Overview

This project implements a comprehensive machine learning pipeline for predicting survival times in a medical context. The dataset contains patient information including demographics, genetic risk factors, and treatment types, with the goal of predicting survival duration while accounting for censored observations.

## Dataset Description

The dataset contains 400 training samples with the following features:

### Features
- **Age**: Patient age (continuous)
- **Gender**: Patient gender (categorical: Male/Female)
- **Stage**: Disease stage (categorical: I, II, III, IV)
- **GeneticRisk**: Genetic risk level (categorical: Low, Medium, High)
- **TreatmentType**: Type of treatment received (categorical: Surgery, Chemotherapy, Radiation, Immunotherapy)
- **ComorbidityIndex**: Comorbidity severity index (continuous)
- **TreatmentResponse**: Response to treatment (categorical: Poor, Partial, Complete)

### Target Variables
- **SurvivalTime**: Time to event or censoring (continuous, in months)
- **Censored**: Censoring indicator (binary: 0 = event observed, 1 = censored)

## Methodology

### Data Preprocessing
- **Missing Data Handling**: The dataset contains no missing values
- **Feature Encoding**: Categorical variables are encoded using appropriate techniques
- **Feature Scaling**: Numerical features are standardized using StandardScaler
- **Train-Test Split**: 80-20 split for model training and evaluation

### Models Implemented

1. **Linear Regression**: Baseline linear model
2. **Polynomial Regression**: Linear regression with polynomial features (degrees 1-6)
3. **Ridge Regression**: L2 regularized linear regression with cross-validation
4. **Lasso Regression**: L1 regularized linear regression with cross-validation
5. **K-Nearest Neighbors (KNN)**: Non-parametric regression with optimized k values (1-29)
6. **CatBoost Regressor**: Gradient boosting with categorical feature support
7. **CatBoost AFT (Accelerated Failure Time)**: Specialized survival analysis model
8. **Histogram-based Gradient Boosting**: Efficient gradient boosting implementation

### Evaluation Metric

The primary evaluation metric is **Censored Mean Squared Error (cMSE)**, which is specifically designed for survival analysis with censored data. This metric accounts for the uncertainty in censored observations by only computing the error for uncensored (observed) events.

### Cross-Validation Strategy

- **K-Fold Cross-Validation**: Tested with 2-10 folds to find optimal configuration
- **Bootstrap Sampling**: Multiple runs with different sample fractions (30%, 50%, 70%) for robust evaluation
- **Hyperparameter Optimization**: Grid search and cross-validation for optimal parameters

## Key Hyperparameters

### Ridge Regression
- **alpha**: Regularization strength (optimized via cross-validation)
- **cv**: Number of cross-validation folds (optimized between 2-10)

### Lasso Regression
- **alpha**: Regularization strength (optimized via cross-validation)
- **cv**: Number of cross-validation folds (optimized between 2-10)

### CatBoost Regressor
- **iterations**: 1000 (number of boosting rounds)
- **learning_rate**: 0.1 (step size for gradient descent)
- **depth**: 6 (maximum tree depth)
- **verbose**: 0 (suppress training output)

### CatBoost AFT Model
- **iterations**: 500 (optimized for survival analysis)
- **learning_rate**: 0.1
- **loss_function**: SurvivalAft with different distributions (Normal, Logistic, Extreme)
- **scale**: Tested values [1.0, 2.0]

### K-Nearest Neighbors
- **n_neighbors**: Optimized range 1-29 through cross-validation
- **weights**: Uniform weighting

### Polynomial Features
- **degree**: Tested polynomial degrees 1-6
- **include_bias**: True

## File Structure

```
Machine Learning Project/
├── README.md                 # Project documentation
├── Script.ipynb             # Main Jupyter notebook with complete analysis
├── Script.py                # Python script version of the notebook
├── train_data.csv           # Training dataset (400 samples)
├── sample_submission.csv    # Sample submission format (100 predictions)
└── requirements.txt         # Python dependencies (if applicable)
```

## Results Summary

The project implements a comprehensive comparison of multiple regression approaches for survival time prediction. The models are evaluated using censored MSE, which appropriately handles the survival analysis context with censored observations.

### Model Performance Highlights
- **Baseline Models**: Linear and polynomial regression provide interpretable baselines
- **Regularized Models**: Ridge and Lasso regression help prevent overfitting
- **Advanced Models**: CatBoost and gradient boosting methods leverage ensemble techniques
- **Survival-Specific**: CatBoost AFT model specifically designed for survival analysis

## Technical Skills Demonstrated

### Machine Learning
- **Supervised Learning**: Regression techniques for continuous target prediction
- **Feature Engineering**: Polynomial features, categorical encoding, feature scaling
- **Model Selection**: Cross-validation, hyperparameter tuning, ensemble methods
- **Survival Analysis**: Handling censored data, specialized loss functions

### Data Science
- **Data Preprocessing**: Missing data analysis, feature transformation
- **Model Evaluation**: Custom metrics for survival analysis, bootstrap validation
- **Visualization**: Data exploration, model performance comparison
- **Statistical Analysis**: Understanding of censoring mechanisms in survival data

### Programming
- **Python**: Pandas, NumPy, Scikit-learn, CatBoost
- **Jupyter Notebooks**: Interactive development and documentation
- **Version Control**: Git-ready project structure
- **Code Quality**: Well-documented, modular code with clear variable naming

## Usage

1. **Environment Setup**: Install required dependencies (pandas, numpy, scikit-learn, catboost, matplotlib, seaborn)
2. **Data Loading**: Load the training data from `train_data.csv`
3. **Model Training**: Run the notebook or script to train all models
4. **Evaluation**: Compare model performance using cMSE metric
5. **Prediction**: Generate predictions for new data using the best-performing model

## Future Improvements

- **Feature Engineering**: Additional domain-specific features, interaction terms
- **Advanced Models**: Deep learning approaches, survival-specific neural networks
- **Hyperparameter Optimization**: Bayesian optimization, automated ML techniques
- **Model Interpretation**: SHAP values, feature importance analysis
- **Validation**: Time-based validation splits, external validation datasets

## Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- catboost
- matplotlib
- seaborn
- jupyter (for notebook execution)

---

*This project demonstrates a thorough approach to survival analysis using modern machine learning techniques, with careful attention to the unique challenges of censored data and appropriate evaluation metrics.*