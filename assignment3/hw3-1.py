import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
# 读取CSV文件
df = pd.read_csv('D:/study material/Grade 2/DDA3020/assignment/assignment3/diabetes.csv')

# 假设您要删除包含零的列名为 'your_column' 的列
column_deal = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 根据零值过滤数据
for i in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    df = df[df[i] != 0]
train_df, test_df = train_test_split(df, test_size=0.33, random_state=33)
X_train = np.array(train_df.iloc[:,0:8])
y_train = np.array(train_df.iloc[:,8]).reshape(-1,1)

X_test  = np.array(test_df.iloc[:,0:8])
y_test  = np.array(test_df.iloc[:,8]).reshape(-1,1)
dt_classifier = tree.DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

y_predict = dt_classifier.predict(X_test)
y_predict = y_predict.reshape(-1,1)
residual = abs(y_test - y_predict)
# print(residual)
error_DT = np.sum(residual)
errors = []
for n_learners in range(20,50):
    base_estimator = tree.DecisionTreeClassifier()
    Bagging = ensemble.BaggingRegressor(base_estimator = base_estimator, n_estimators = n_learners)
    y_train = y_train
    Bagging.fit(X_train,y_train.ravel())
    y_predict = Bagging.predict(X_test)
    y_predict = y_predict.reshape(-1,1)
    residual = abs(y_test - y_predict)
    # print(residual)
    error_BG = np.sum(residual)
    errors.append(error_BG)
errors = np.array(errors)
x = np.array(range(20,50))
error_DT = np.array([error_DT]*30)
figure,ax = plt.subplots()
