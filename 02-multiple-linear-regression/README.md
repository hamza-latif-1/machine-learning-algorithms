# Multiple Linear Regression From Scratch

A Python implementation of **Multiple Linear Regression** using NumPy and the Ordinary Least Squares (OLS) Normal Equation.

The objective of this project is to understand how Multiple Linear Regression works mathematically and implement it from scratch without using Scikit-learn for model training.

## 1. Mathematical Formulation

Multiple Linear Regression predicts a continuous target variable using two or more independent variables (features).

The prediction equation is:

$$
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_px_p
$$

Where:

- $\hat{y}$ = predicted target value
- $\beta_0$ = intercept (bias)
- $\beta_1, \beta_2, \ldots, \beta_p$ = coefficients (weights)
- $x_1, x_2, \ldots, x_p$ = independent variables (features)
- $p$ = number of features

Each coefficient represents the expected change in the prediction when its corresponding feature increases by one unit, while the other features remain constant.

### Matrix Representation

The prediction equation can be represented in matrix form:

$$
\hat{y} = X\beta
$$

To include the intercept, we add a column of ones to the original feature matrix:

$$
X_b =
\begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p}\\
1 & x_{21} & x_{22} & \cdots & x_{2p}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
1 & x_{n1} & x_{n2} & \cdots & x_{np}
\end{bmatrix}
$$

The coefficient vector becomes:

$$
\beta =
\begin{bmatrix}
\beta_0\\
\beta_1\\
\beta_2\\
\vdots\\
\beta_p
\end{bmatrix}
$$

Therefore, the complete prediction equation is:

$$
\hat{y} = X_b\beta
$$

### Ordinary Least Squares (OLS)

OLS finds the coefficients that minimize the sum of squared differences between actual and predicted values.

The cost function is:

$$
J(\beta) = (y-X_b\beta)^T(y-X_b\beta)
$$

To minimize this cost function, we differentiate with respect to $\beta$ and set the derivative equal to zero:

$$
\frac{\partial J(\beta)}{\partial \beta}
=
-2X_b^T(y-X_b\beta)=0
$$

Rearranging:

$$
X_b^TX_b\beta = X_b^Ty
$$

Solving for the coefficients gives the **Normal Equation**:

$$
\boxed{
\hat{\beta}=(X_b^TX_b)^{-1}X_b^Ty
}
$$

This equation calculates the optimal coefficients directly, without iterative optimization such as Gradient Descent.

The inverse-based expression requires $X_b^TX_b$ to be invertible.

## 2. Implementation Using NumPy

The model is implemented using a Python class named `MyLinearRegression`.

### Step 1: Initialize Parameters

```python
self.coef_ = None
self.intercept_ = None
```

These attributes store the learned coefficients and intercept.

### Step 2: Add the Intercept Column

```python
X_train = np.insert(X_train, 0, 1, axis=1)
```

A column of ones is added to the feature matrix so that the intercept can be learned as part of the coefficient vector.

### Step 3: Calculate the Coefficients

```python
beta = np.linalg.inv(
    np.dot(X_train.T, X_train)
).dot(X_train.T).dot(y_train)
```

This implements the Normal Equation:

$$
\hat{\beta}=(X_b^TX_b)^{-1}X_b^Ty
$$

### Step 4: Separate the Intercept and Weights

```python
self.intercept_ = beta[0]
self.coef_ = beta[1:]
```

The first element represents the intercept, while the remaining elements represent the feature coefficients.

### Step 5: Make Predictions

```python
y_pred = np.dot(X_test, self.coef_) + self.intercept_
```

Predictions are calculated using matrix multiplication between the test feature matrix and the learned coefficients, followed by adding the intercept.

## 3. Dataset and Experiment

The model was tested using the **Diabetes dataset** available through Scikit-learn.

```python
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
```

Dataset characteristics:

| Property | Value |
|---|---|
| Total samples | 442 |
| Number of features | 10 |
| Target | Quantitative disease progression measure |
| Task | Regression |

The dataset was divided into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
```

## 4. Comparison
