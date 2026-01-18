# AML_Project | COMS-W4995 | Applied Machine Learning Project

# 🧠 Using Machine Learning for Predicting Sleep Health & Lifestyle Patterns

**Daniel David, Zhangjing Cheng, Cooper Orio, Yuxuan Chen**  
*Supervised by Dr. Vijay Pappu – Columbia University – May 19, 2025*

---

## 📘 Project Overview

This project investigates how machine learning can be used to predict sleep quality based on lifestyle and physiological features. Using the Sleep Health and Lifestyle Dataset from Kaggle, we evaluate four key models: **Random Forest**, **XGBoost**, **Support Vector Machines (SVM)**, and **Neural Networks**, with a focus on handling **class imbalance**, analyzing **feature importance**, and optimizing for **average precision** (AP) as well as accuracy.

---

## 🛠️ Tools & Technologies

- **Python** (pandas, numpy, seaborn, matplotlib, scikit-learn, imbalanced-learn)
- **Jupyter Notebook / Google Colab**
- **XGBoost**
- **SMOTE** (Synthetic Minority Oversampling Technique)
- **GridSearchCV** with Stratified K-Fold cross-validation
- **Matplotlib & Seaborn** for visualizations

---

## 🧪 Methodology

- **Data Preprocessing**: Handled missing values, removed duplicate records, decomposed blood pressure into systolic/diastolic, and encoded categorical variables.
- **Target Definition**: Transformed sleep quality scores into three categories – *Poor*, *Moderate*, and *Good*.
- **Class Imbalance Handling**: Applied SMOTE to oversample minority classes during training.
- **Model Training & Tuning**: Used `GridSearchCV` to find optimal hyperparameters for each model, with both **accuracy** and **average precision** as scoring metrics.
- **Evaluation**: Used classification reports, confusion matrices, and feature importance plots to compare model performance.

---

## 📈 Results Summary

- **Random Forest**: Achieved up to **97% accuracy** on the deduplicated dataset, with top features including stress level, daily steps, and heart rate.
- **XGBoost**: Delivered **93% accuracy** when tuned for average precision. Prioritized daily steps and stress level.
- **Neural Network**: Reached **90% accuracy**, though more sensitive to class imbalance.
- **SVM (One-vs-Rest)**: Performed competitively with **90–93% accuracy**, especially with a linear kernel.

Across all models, **stress level**, **daily steps**, **sleep duration**, and **occupation** consistently ranked as the most influential predictors.


---

## 📚 Resources

- **📑 Colab Notebook**: [Sleep ML Colab](https://colab.research.google.com/drive/1F7Oqvgn6pYPArlhtxbbzQpmVsxkB1_1h)
- **🗃️ Dataset**: [Kaggle - Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

---

## 🔍 Key Learnings

- Optimizing for **average precision** significantly improves minority class detection.
- Feature importance varies across models, but physiological and behavioral factors dominate.
- Deduplication prevents data leakage and improves generalization.

---

## 📌 Citation

If you use this repository or its findings in your work, please cite:

> Daniel David, Zhangjing Cheng, Cooper Orio, Yuxuan Chen. *Using Machine Learning for Predicting Sleep Health & Lifestyle Patterns*. Columbia University, 2025.
