import numpy as np
import matplotlib.pyplot as plt
import csv
data = []
with open('D:\study material\Grade 2\DDA3020\\assignment\Regression.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        data.append(list(map(float, row[2:])))
data = np.array(data)
nan_rows = np.isnan(data).any(axis=1)
data = np.delete(data, nan_rows, axis=0)
learning_rate = 0.1
iterations = 1000
num_trials = 10
# (trial, X_train, y_train, X_test, y_test, coefficients)
seeds = []
RMSE_train = []
RMSE_test = []
trials = []
for trial in range(num_trials):
    # 1. shuffle the data randomly and split the data
    np.random.shuffle(data)
    split_index = int(0.8 * len(data))
    training_data, testing_data = data[:split_index], data[split_index:]

    # 2. normalization
    X_train, y_train = training_data[:, :21], training_data[:, 21:]
    X_test, y_test = testing_data[:, :21], testing_data[:, 21:]

    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    X_train = (X_train - min_vals) / (max_vals - min_vals)

    min_vals = y_train.min(axis=0)
    max_vals = y_train.max(axis=0)
    y_train = (y_train - min_vals) / (max_vals - min_vals)

    min_vals = X_test.min(axis=0)
    max_vals = X_test.max(axis=0)
    X_test  = (X_test - min_vals) / (max_vals - min_vals)

    min_vals = y_test.min(axis=0)
    max_vals = y_test.max(axis=0)
    y_test  = (y_test - min_vals) / (max_vals - min_vals)


    # Gradient-Descent
    def linear_regression_fit(X, y, learning_rate, iterations):
        theta = np.random.randn(21, 2)
        m = len(y)
        for iteration in range(iterations):
            y_pred = X.dot(theta)
            gradient = (1 / m) * X.T.dot(y_pred - y)
            theta -= learning_rate * gradient
        return theta

    def linear_regression_predict(X, coefficients):
        return X @ coefficients

    # 3. train the model
    coefficients = linear_regression_fit(X_train, y_train,learning_rate,iterations)

    # 4. make predictions
    predictions1 = linear_regression_predict(X_train, coefficients)
    predictions2 = linear_regression_predict(X_test, coefficients)

    # Calculate RMSE
    seeds.append((trial, X_train, y_train, X_test, y_test, coefficients))
    trials.append(trial+1)
    rmse_train = np.sqrt(np.mean((predictions1 - y_train) ** 2))
    rmse_test = np.sqrt(np.mean((predictions2 - y_test) ** 2))
    RMSE_train.append(rmse_train)
    RMSE_test.append(rmse_test)
    print('Trial: '+ str(trial + 1) + '  Traing RMSE: '+str(rmse_train)+'  Testing RMSE: '+str(rmse_test))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(trials,RMSE_train, color="blue", label = 'train'  )
ax.plot(trials,RMSE_test , color='red', label = 'test')