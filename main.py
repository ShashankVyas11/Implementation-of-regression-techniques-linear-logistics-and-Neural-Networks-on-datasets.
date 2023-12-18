# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression, make_classification
from sklearn.neural_network import MLPRegressor, MLPClassifier

# Generate a synthetic regression dataset
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split the dataset into training and testing sets
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Linear Regression
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = linear_reg_model.predict(X_reg_test)

# Visualize the linear regression
plt.scatter(X_reg_test, y_reg_test, color='black')
plt.plot(X_reg_test, y_reg_pred, color='blue', linewidth=3)
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Evaluate the linear regression model
regression_mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f'Linear Regression MSE: {regression_mse}')

# Generate a synthetic classification dataset
X_cls, y_cls = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

# Split the dataset into training and testing sets
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Logistic Regression
logistic_reg_model = LogisticRegression()
logistic_reg_model.fit(X_cls_train, y_cls_train)
y_cls_pred = logistic_reg_model.predict(X_cls_test)

# Visualize the logistic regression
plt.scatter(X_cls_test[:, 0], X_cls_test[:, 1], c=y_cls_test, cmap=plt.cm.Paired, edgecolors='k')
plt.title('Logistic Regression')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Evaluate the logistic regression model
classification_accuracy = accuracy_score(y_cls_test, y_cls_pred)
print(f'Logistic Regression Accuracy: {classification_accuracy}')

# Neural Network for Regression
scaler = StandardScaler()
X_reg_train_scaled = scaler.fit_transform(X_reg_train)
X_reg_test_scaled = scaler.transform(X_reg_test)

nn_reg_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
nn_reg_model.fit(X_reg_train_scaled, y_reg_train)
y_reg_nn_pred = nn_reg_model.predict(X_reg_test_scaled)

# Visualize the neural network regression
plt.scatter(X_reg_test, y_reg_test, color='black')
plt.scatter(X_reg_test, y_reg_nn_pred, color='red', marker='x')
plt.title('Neural Network Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Evaluate the neural network regression model
nn_regression_mse = mean_squared_error(y_reg_test, y_reg_nn_pred)
print(f'Neural Network Regression MSE: {nn_regression_mse}')

# Neural Network for Classification
scaler_cls = StandardScaler()
X_cls_train_scaled = scaler_cls.fit_transform(X_cls_train)
X_cls_test_scaled = scaler_cls.transform(X_cls_test)

nn_cls_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
nn_cls_model.fit(X_cls_train_scaled, y_cls_train)
y_cls_nn_pred = nn_cls_model.predict(X_cls_test_scaled)

# Visualize the neural network classification
plt.scatter(X_cls_test[:, 0], X_cls_test[:, 1], c=y_cls_test, cmap=plt.cm.Paired, edgecolors='k')
plt.scatter(X_cls_test[:, 0], X_cls_test[:, 1], c=y_cls_nn_pred, cmap=plt.cm.Paired, marker='x')
plt.title('Neural Network Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Evaluate the neural network classification model
nn_classification_accuracy = accuracy_score(y_cls_test, y_cls_nn_pred)
print(f'Neural Network Classification Accuracy: {nn_classification_accuracy}')
