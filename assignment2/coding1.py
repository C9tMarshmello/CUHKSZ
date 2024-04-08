import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, num_features):
        self.weights = np.zeros((num_features))
        self.bias = 0

    def gradient_descent(self, X, y):
        m, n = X.shape
        self.initialize_parameters(n)

        for i in range(self.num_iterations):
            # Complete your implementation here.
            # Forward pass (sigmoid function provided)
            z = X @ self.weights.T + self.bias
            fx = self.sigmoid(z)
            # Compute cost
            cost = -1/m * np.sum(y * np.log(fx) + (1 - y) * np.log(1 - fx))
            # Compute gradients
            # gradient = 0
            # for i in range(len(X)):
            #     gradient += (fx[i]-y[i])*X[i]
            print(np.hstack(((fx-y).T @ X[:,0:1],(fx-y).T @ X[:,1:])))
            dw = 1/m * np.hstack(((fx-y).T @ X[:,0:1],(fx-y).T @ X[:,1:]))
            db = np.sum(fx-y)
            # Update parameters
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            # Print the cost every 100 iterations
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        fx = self.sigmoid(z)
        predictions = (fx > 0.5).astype(int)
        return predictions



# Generate some example data
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Instantiate and train the logistic regression model
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.gradient_descent(X, y)

# Predict and output accuracy
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy}")
