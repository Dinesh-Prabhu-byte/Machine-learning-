import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36])

# ---------------- Linear Regression ----------------
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)
mse_linear = mean_squared_error(y, y_pred_linear)

# ---------------- Polynomial Regression (degree = 2) ----------------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)
mse_poly = mean_squared_error(y, y_pred_poly)

# Results
print("Linear Regression Predictions:", y_pred_linear)
print("Linear Regression MSE:", mse_linear)

print("\nPolynomial Regression Predictions:", y_pred_poly)
print("Polynomial Regression MSE:", mse_poly)
