# Customer Churn & Price Sensitivity Analyzer

## Overview
A binary classification and data analytics system designed to predict customer subscription cancellation risk, optimize recall metrics for high-risk user detection, and quantify price sensitivity thresholds.

## Key Highlights
* **Preprocessing & Feature Scaling:** Preprocessed user interaction logs, encoded categorical variables via One-Hot Encoding, and applied `StandardScaler` to continuous metrics.
* **Class Balancing (SMOTE):** Integrated Synthetic Minority Over-sampling Technique (SMOTE) to rebalance severe majority-minority class distributions without model overfitting.
* **Recall Optimization:** Tuned classification thresholds across Logistic Regression and Random Forest models to maximize Recall (high detection of at-risk churners).
* **Price Elasticity Analysis:** Derived feature weights from Logistic Regression coefficients to isolate price-hike variables and pinpoint exact price limits where customer cancellations sharply spike.
* **Results:** Achieved an 82% Recall score on churn identification and established data-backed pricing boundaries for customer retention strategies.

## Tech Stack & Tools
* **Language:** Python
* **Libraries:** Scikit-Learn, Imbalanced-Learn (SMOTE), Pandas, NumPy
* **Concepts:** Binary Classification, Hyperparameter Tuning, Price Elasticity, Recall Optimization
