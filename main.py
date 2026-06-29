import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def main(url):

    df = pd.read_csv(url)

    #print(df.head())
    #df.info()

    #print(df.describe())
    #print(df.isnull().sum())

    #there is no missing value after checking
    df_num = df.select_dtypes(include=['int64', 'float64'])
    df_cal = df.select_dtypes(exclude=['int64', 'float64']).drop(columns=["Attrition"])


    #plot for numeric values
    for col in df_num.columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(y=df[col],x=df["Attrition"])
        plt.title(f"Boxplot of {col}")
        # plt.show()
        plt.close()

    for col in df_num.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col],kde=True)
        plt.title(f"Histplot of {col}")
        # plt.show()
        plt.close()

    for col in df_cal.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(x=df[col],hue=df["Attrition"])
        plt.title(f"Countplot of {col}")
        # plt.show()
        plt.close()

    attrition_dummy = pd.get_dummies(df["Attrition"],drop_first=True,dtype=int)



    df_final=df.drop(columns=["EmployeeNumber","EmployeeCount","Over18",'StandardHours','Attrition'])
    for col in df_cal.columns:
        hot_one=pd.get_dummies(df[col],drop_first=True,dtype=int)
        df_final=pd.concat([df_final,hot_one],axis=1)


    df_final=df_final.select_dtypes(exclude="object")
    #print(df_final.describe())
    #print(df_final.info())
    Y=df_final["Yes"]

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(df_final.drop(columns="Yes", axis=1), Y, train_size=0.8,
                                                                random_state=3)
    # we need to use LogisticRegression because our target value is categorical
    feature_names = x_train_raw.columns
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)

    logistic=LogisticRegression(max_iter=1000,class_weight="balanced")
    logistic.fit(x_train,y_train)

    ythat=logistic.predict(x_test)

    print(classification_report(y_test, ythat))
    coe=logistic.coef_

    print(df_final.columns[df_final.columns.duplicated()])
    print("Baseline (predict majority):", (y_test == 0).mean())

    print("Accuracy:", accuracy_score(y_test, ythat))
    print("Confusion Matrix:\n", confusion_matrix(y_test, ythat))


    for feat, c in sorted(zip(feature_names, coe[0]), key=lambda t: abs(t[1]), reverse=True):
        print(f"{feat}: {c:.3f}")

    df_box=df_final[["JobLevel","TotalWorkingYears","YearsInCurrentRole","Age","MonthlyIncome"]]
    for col in df_box.columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(x = df_final["Yes"],y = df_box[col])
        plt.title(f"boxPlot of {col}")
        # plt.show()
        plt.close()
#note: employees leave who have fewer working years
# Younger employees tend to stay longer, while older employees may leave due to retirement planning or career transitions.
# employees who have higher salary might like to leave
#  Suggests that salary alone does not guarantee retention. Factors such as job satisfaction, stress, or career growth opportunities may influence their decision to leave.

    df_save = x_test_raw.copy()
    df_save["Attrition predicted values"] = ythat
    df_save["Actual values"] = y_test.values
    df_save.to_csv("attrition_prediction.csv", index=False)





main("/Users/amirali/PycharmProjects/PythonProject/Employees-Attrition-prediction/WA_Fn-UseC_-HR-Employee-Attrition.csv")


