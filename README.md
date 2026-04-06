# Sleep Quality Prediction using Lifestyle Factors

This project implements machine learning models to predict 'Quality of Sleep' based on various lifestyle factors. The goal is to perform multi-class classification using Logistic Regression and K-Nearest Neighbors (KNN) algorithms. The notebook covers data loading, extensive preprocessing, model training, and thorough evaluation.

## Table of Contents

1.  [Data Loading and Initial Exploration](#1.-Data-Loading-and-Initial-Exploration)
2.  [Data Cleaning and Preprocessing](#2.-Data-Cleaning-and-Preprocessing)
3.  [Feature Engineering](#3.-Feature-Engineering)
4.  [Train-Test Split and Feature Scaling](#4.-Train-Test-Split-and-Feature-Scaling)
5.  [Model Training and Evaluation - Logistic Regression](#5.-Model-Training-and-Evaluation---Logistic-Regression)
6.  [Model Training and Evaluation - K-Nearest Neighbors (KNN)](#6.-Model-Training-and-Evaluation---K-Nearest-Neighbors-(KNN))
7.  [Model Comparison and User Prediction Function](#7.-Model-Comparison-and-User-Prediction-Function)
8.  [Additional Visualization: ROC Curves](#8.-Additional-Visualization:-ROC-Curves)

## 1. Data Loading and Initial Exploration

The project begins by loading the `Sleep_health_and_lifestyle_dataset.csv` dataset and performing initial checks to understand its structure, identify missing values, and analyze descriptive statistics. This includes:

-   Importing necessary libraries (pandas, numpy, matplotlib, seaborn).
-   Mounting Google Drive to access the dataset.
-   Displaying the first few rows (`data.head()`).
-   Checking data types and non-null counts (`data.info()`).
-   Reviewing statistical summaries (`data.describe()`).
-   Identifying null values (`data.isnull().sum()`).
-   Checking for duplicate rows.
-   Separating numerical and categorical features.

## 2. Data Cleaning and Preprocessing

This section focuses on preparing the data for modeling, including:

-   Dropping irrelevant features such as 'Person ID'.
-   Splitting the 'Blood Pressure' column into 'Systolic' and 'Diastolic' components and converting them to integer types, then dropping the original 'Blood Pressure' column.
-   Handling categorical data through encoding techniques.

## 3. Feature Engineering

-   **Label Encoding 'Gender'**: Converting 'Gender' into numerical representation (Male: 1, Female: 0).
-   **One-Hot Encoding**: Applying one-hot encoding to 'Occupation' and 'BMI Category' to create dummy variables, removing the first category to avoid multicollinearity.
-   **Target Variable Encoding**: Filling NaN values in 'Sleep Disorder' with 'None' and then label encoding it into a multi-class numerical target ('None': 0, 'Insomnia': 1, 'Sleep Apnea': 2).

## 4. Train-Test Split and Feature Scaling

The data is prepared for model training and evaluation:

-   **Feature and Target Definition**: `X` is defined as features (dropping 'Quality of Sleep', 'Sleep Disorder', and 'Sleep Disorder Encoded' to avoid data leakage for the sleep disorder prediction task), and `y` is the 'Sleep Disorder Encoded' target.
-   **Train-Test Split**: The dataset is split into training (75%) and testing (25%) sets using `train_test_split`, with `stratify=y` to maintain the class distribution in both sets.
-   **Handling Imbalanced Data (SMOTE)**: SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to address class imbalance in the target variable, ensuring that minority classes are adequately represented.
-   **Feature Scaling**: Numerical features are scaled using `StandardScaler` to normalize their ranges. The scaler is fitted on the training data and then used to transform both training and testing sets.

## 5. Model Training and Evaluation - Logistic Regression

-   **Model Initialization**: A `LogisticRegression` model is initialized with `max_iter=2000`, `C=0.5`, `solver='lbfgs'`, `multi_class='multinomial'`, and `class_weight='balanced'` to handle multi-class classification and class imbalance.
-   **Training**: The model is trained on the scaled training data (`X_train_scaled`, `y_train`).
-   **Prediction**: Predictions are made on both training and testing sets.
-   **Evaluation Metrics**: Performance is assessed using `accuracy_score`, `precision_score`, `recall_score`, and `f1_score` for both train and test data, with `average='weighted'` for multi-class metrics.
-   **Confusion Matrix**: A confusion matrix is generated and displayed to visualize the model's classification performance.

## 6. Model Training and Evaluation - K-Nearest Neighbors (KNN)

-   **Optimal k Selection**: A loop iterates through various `k` values (1 to 20) to find the optimal number of neighbors by plotting training and testing accuracies.
-   **Model Initialization and Training**: A `KNeighborsClassifier` model is initialized with the `best_k` (determined to be 13) and trained on the scaled training data.
-   **Prediction**: Predictions are made on both training and testing sets.
-   **Evaluation Metrics**: Similar to Logistic Regression, performance is evaluated using `accuracy_score`, `precision_score`, `recall_score`, and `f1_score` for both train and test data.
-   **Confusion Matrix**: A confusion matrix is generated and displayed for KNN.

## 7. Model Comparison and User Prediction Function

-   **Metric Comparison**: A helper function `evaluate_model` is used to print and compare accuracy, precision, recall, and F1-score for both models on the train and test datasets.
-   **Visual Comparison**: 
    -   Confusion matrices for both models on the test set are displayed side-by-side using heatmaps for easy comparison.
    -   Classification reports (precision, recall, f1-score per class) are printed for both models.
    -   A DataFrame `results_df` is created to tabularize the test performance metrics (Accuracy, Precision, Recall, F1 Score) for both models.
    -   Bar charts are generated to visually compare the overall accuracy and other metrics of the two models.
    -   Heatmaps of the classification reports provide a per-class metric comparison.

## 8. Additional Visualization: ROC Curves

-   **One-vs-Rest ROC Curves**: ROC (Receiver Operating Characteristic) curves are plotted for both Logistic Regression and KNN models for each class using the one-vs-rest approach. This helps visualize the trade-off between the true positive rate and the false positive rate at various thresholds.
-   **AUC Calculation**: The Area Under the Curve (AUC) is calculated for each class, providing a single metric to summarize the ROC curve, indicating the model's discriminative power.

This notebook provides a comprehensive analysis of sleep quality prediction using two common machine learning algorithms, highlighting their performance through various evaluation metrics and visualizations.
