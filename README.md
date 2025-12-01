Employee Attrition Prediction
Project Overview
This project aims to predict employee attrition using the IBM HR Analytics dataset. Attrition refers to employees leaving a company, and understanding its drivers helps HR departments retain talent and make better workforce decisions.
The project demonstrates data preprocessing, exploratory data analysis (EDA), feature engineering, and classification modeling using Python and Scikit-Learn.
Dataset
The dataset is taken from IBM HR Analytics – Employee Attrition Dataset. It contains 1470 employee records with features including:
Demographics (Age, Gender, Marital Status)
Job details (Job Role, Department, Job Level, Salary)
Work history (Years at Company, Years in Current Role, Total Working Years)
Performance indicators (Performance Rating, Work-Life Balance)
Target variable: Attrition (Yes / No)
Key Steps
1. Data Exploration & Visualization
Checked dataset structure, data types, and missing values
Visualized numerical features with histograms and boxplots
Visualized categorical features with countplots
Investigated correlations between features and attrition
2. Data Cleaning & Preprocessing
Removed irrelevant columns (EmployeeNumber, EmployeeCount, Over18)
Encoded categorical variables using one-hot encoding
Ensured all features were numeric for modeling
3. Modeling
Used Logistic Regression to predict the categorical target
Split dataset into train and test sets (80% train / 20% test)
Balanced the model to account for class imbalance
4. Model Evaluation
Accuracy Score: Measures overall correct predictions
Confusion Matrix: Shows True Positives, True Negatives, False Positives, False Negatives
Feature Insights: Boxplots revealed patterns such as:
Younger employees tend to stay longer
Employees with higher salary might leave
Employees with fewer working hours may leave
5. Predictions
Predicted attrition on the test set
Saved predictions along with actual values to attrition_prediction.csv
Insights
Salary alone does not guarantee retention; other factors such as job satisfaction, work-life balance, and career growth opportunities play a crucial role.
Age and experience influence attrition patterns. Older employees may leave due to career transitions or retirement planning.
Job level and tenure provide insights into employee commitment and risk of attrition.
