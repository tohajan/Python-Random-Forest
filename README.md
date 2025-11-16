# Depression Classification Using Random Forests  
### Handling Class Imbalance with Class Weights and SMOTE

This project explores the use of supervised machine learning to predict depressive symptoms (“Yes” vs. “No”) using a survey-based dataset (National Helath Interview Survey[NHIS] 2021). The notebook walks through a full applied ML workflow in Python, including data preprocessing, model training, evaluation, and class-imbalance mitigation techniques.

The primary model is a Random Forest classifier implemented in `scikit-learn`, with multiple strategies tested to improve minority-class performance.

---

## Project Objectives
- Build a reproducible classification pipeline for a binary mental-health outcome.
- Explore class imbalance challenges and evaluate multiple solutions:
  - Baseline Random Forest
  - `class_weight='balanced'` adjustments
  - SMOTE oversampling using `imblearn`
- Assess model performance using accuracy, precision/recall, F1, confusion matrices, ROC-AUC, and PR-AUC.
- Provide transparent interpretation of what works, what doesn’t, and why.

---

## Repository Structure

├── Predicting depression_RandomForest.ipynb # Main notebook
├── data/ # Source dataset
└── README.md # Overview + documentation