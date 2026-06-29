# Employee Attrition Prediction

## Overview

This project predicts employee attrition (whether an employee leaves a company) using the IBM HR Analytics dataset. Understanding the drivers of attrition helps HR teams identify at-risk employees and make better retention decisions. The project covers data preprocessing, exploratory data analysis (EDA), feature engineering, and classification with logistic regression.

## Dataset

[IBM HR Analytics – Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — 1,470 employee records.

Features include:
- **Demographics**: Age, Gender, Marital Status
- **Job details**: Job Role, Department, Job Level, Monthly Income
- **Work history**: Years at Company, Years in Current Role, Total Working Years
- **Satisfaction / performance**: Job Satisfaction, Work-Life Balance, Performance Rating
- **Target**: Attrition (Yes / No)

The classes are imbalanced — roughly 84% "No" and 16% "Yes" — which is central to how the results below should be read.

## Steps

**1. Exploration & visualization**
Checked structure, data types, and missing values (none found). Visualized numeric features with histograms and boxplots, and categorical features with countplots, against the attrition target.

**2. Cleaning & preprocessing**
Dropped non-predictive columns (`EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours` — the last two have zero variance). One-hot encoded categorical variables (`drop_first=True`) and standardized numeric features with `StandardScaler` (fit on the training set only).

**3. Modeling**
Logistic regression with `class_weight="balanced"` to account for the class imbalance. 80/20 train/test split.

**4. Evaluation**

| Metric | Value |
|---|---|
| Baseline (always predict "No") | 71.4% |
| Model accuracy | 51.7% |
| Recall on leavers (class "Yes") | 0.54 |
| Recall on stayers (class "No") | 0.51 |

Confusion matrix (test set, 294 employees):

```
              Predicted No   Predicted Yes
Actual No         107            103
Actual Yes         39             45
```

**Reading these results honestly:** the balanced logistic model scores *below* the 71.4% majority-class baseline on raw accuracy. But accuracy is misleading here — a model that simply predicts "No" for everyone hits 71% while catching **zero** leavers, making it useless for the actual goal of *identifying at-risk employees*. The balanced model deliberately trades overall accuracy to catch ~54% of employees who actually leave. For comparison, a Random Forest tested on the same data reached 71% accuracy but collapsed to predicting the majority class for everyone (0% recall on leavers) — a higher number that is worse for the task. This project illustrates the precision/recall tradeoff in imbalanced classification, where the right metric depends on the business goal, not just headline accuracy.

**5. Feature insights**
The strongest standardized coefficients pointed to `MonthlyIncome`, `JobLevel`, business travel frequency, `TrainingTimesLastYear`, and `Age` as the most influential features. Boxplots of tenure and income against attrition suggested patterns worth noting: employees with fewer total working years left more often, and higher salary alone did not guarantee retention — consistent with satisfaction, work-life balance, and career growth mattering alongside pay.

Test-set predictions alongside actual values are saved to `attrition_prediction.csv`.

## Tech stack

Python, Pandas, Scikit-Learn, Matplotlib, Seaborn

## Run

```bash
pip install -r requirements.txt
python main.py
```

Update the dataset path in the `main(...)` call at the bottom of `main.py` to point to your local copy of the CSV.

## Limitations & next steps

- The IBM HR dataset is known to be difficult to separate with linear models; accuracy below baseline is expected for plain logistic regression.
- A properly tuned tree-based model (with depth limits and threshold adjustment) would likely improve minority-class recall — a natural next step.