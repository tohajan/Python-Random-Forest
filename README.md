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

## Methods Overview
- **Language/Stack:** Python, Jupyter Notebook  
- **Core Libraries:**  
  - `pandas`, `numpy` for data handling  
  - `scikit-learn` for modeling + evaluation  
  - `imblearn` for SMOTE  
  - `matplotlib` & `seaborn` for visualization  
- **Model:** Random Forest Classifier  
- **Evaluation:** Confusion matrix, classification report, ROC-AUC, PR-AUC

---

## Summary of Findings
- Baseline Random Forest achieved strong performance for the majority class but weak recall for the minority (“Yes”) class.
- Adding `class_weight='balanced'` did not substantially improve minority-class detection.
- SMOTE increased recall but did not meaningfully improve precision or overall discriminative ability.
- Results suggest that *feature limitations*, rather than algorithm choice, are the primary bottleneck.

A detailed narrative interpretation of these findings is included at the end of the notebook.

---

## Future Work
- Try gradient boosting models (XGBoost, LightGBM).
- Apply threshold-moving to improve recall.
- Explore domain-informed feature engineering.
- Collect richer predictive features.

---

## License
This project is released for educational and portfolio purposes.  
Feel free to fork and adapt.



## Repository Structure

├── data/ # Source dataset  
├── Predicting depression_RandomForest.ipynb # Main notebook  
└── README.md # Overview + documentation
