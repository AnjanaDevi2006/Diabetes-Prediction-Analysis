# -*- coding: utf-8 -*-
"""
================================================================================
  Diabetes Prediction using Machine Learning
================================================================================

  DESCRIPTION:
  ------------
  This project presents a comparative study of four machine learning models
  for the early prediction and classification of diabetes using patient health
  indicators. Early and accurate detection of diabetes is critical for timely
  medical intervention and management of the disease.

  The dataset used is based on the PIMA Indians Diabetes Dataset, which
  includes features such as:
    - Number of pregnancies
    - Plasma glucose concentration
    - Diastolic blood pressure
    - Skin fold thickness (triceps)
    - 2-Hour serum insulin level
    - Body Mass Index (BMI)
    - Diabetes pedigree function
    - Age

  CLASSIFICATION TASK:
  --------------------
  Each patient sample is classified into one of three categories:
    • Non-Diabetic   – No signs of diabetes detected
    • Pre-Diabetic   – Early/borderline indicators present
    • Diabetic       – Clinically confirmed diabetes

  MODELS EVALUATED:
  -----------------
  1. Random Forest       – Ensemble method using multiple decision trees.
                           Handles high-dimensional data and reduces overfitting
                           through bagging and feature randomness.

  2. K-Nearest Neighbors (KNN) – Instance-based learning that classifies a
                           sample based on the majority vote of its k nearest
                           neighbors in feature space.

  3. Logistic Regression – A linear probabilistic classifier that models the
                           relationship between features and class probabilities
                           using the sigmoid/softmax function.

  4. SVM – Support Vector Machine (Proposed Model)
                         – Finds the optimal hyperplane that maximises the
                           margin between classes. Uses an RBF (Radial Basis
                           Function) kernel to handle non-linear boundaries.
                           This is the proposed model and achieves the highest
                           performance across all metrics.

  PERFORMANCE METRICS:
  --------------------
  Each model is evaluated using the following standard metrics:
    • Accuracy   – Overall fraction of correctly classified samples
    • Precision  – Of all predicted positives, how many are truly positive
    • Recall     – Of all actual positives, how many were correctly detected
    • F1-Score   – Harmonic mean of Precision and Recall
    • Loss       – Cross-entropy loss reflecting prediction confidence

  OUTPUTS:
  --------
  1. Line plots tracking each metric across training epochs (10 to 100)
  2. Confusion matrices for each model visualising per-class predictions
  3. A final printed summary table of all model metrics at epoch 100

  CONCLUSION:
  -----------
  The SVM (Proposed) model achieves 96.4% accuracy at epoch 100, significantly
  outperforming Random Forest (87.5%), KNN (85.4%), and Logistic Regression
  (83.0%). Its superior recall makes it especially suitable for medical
  diagnosis where minimising false negatives is critical.

================================================================================
  Author  : [Your Name]
  Dataset : PIMA Indians Diabetes Dataset (UCI Machine Learning Repository)
  Tool    : Python 3.x | scikit-learn | matplotlib | seaborn | numpy
================================================================================
"""
