# Gradient Descent for Multiple Linear Regression

## Overview

This project demonstrates the implementation of **Gradient Descent for Multiple Linear Regression from scratch using NumPy**.

The objective is to understand how gradient descent optimizes multiple model parameters (coefficients and intercept) to minimize the Mean Squared Error (MSE) and compare the implementation with Scikit-learn's Linear Regression model.

## Dataset

The implementation uses the **Diabetes dataset** available through Scikit-learn.

- **Observations:** 442
- **Features:** 10
- **Training samples:** 353
- **Testing samples:** 89
- **Train-test split:** 80:20

## Mathematical Foundation

### 1. Hypothesis Function

For multiple linear regression, the predicted output is:

$$
\hat{y}=w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

Where:

- $w_1,w_2,\ldots,w_n$ are model coefficients.
- $x_1,x_2,\ldots,x_n$ are input features.
- $b$ is the intercept.
- $\hat{y}$ is the predicted output.

### 2. Cost Function (Mean Squared Error)

The Mean Squared Error measures the difference between actual and predicted values.

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2
$$

Where:

- $m$ is the number of training examples.
- $\hat{y}^{(i)}$ is the predicted value.
- $y^{(i)}$ is the actual value.

### 3. Gradient Descent

Gradient descent updates model parameters iteratively to minimize the cost function.

**Gradient with respect to each coefficient:**

$$
\frac{\partial J}{\partial w_j}
$$
=
$$
\frac{2}{m}\sum_{i=1}^{m}
(\hat{y}^{(i)}-y^{(i)})x_j^{(i)}
$$

**Gradient with respect to the intercept:**

$$
\frac{\partial J}{\partial b}
$$
=
$$
\frac{2}{m}\sum_{i=1}^{m}
(\hat{y}^{(i)}-y^{(i)})
$$

**Coefficient update:**

$$
w_j=w_j-\alpha\frac{\partial J}{\partial w_j}
$$

**Intercept update:**

$$
b=b-\alpha\frac{\partial J}{\partial b}
$$

Where $\alpha$ represents the learning rate.

## Implementation

A custom `GDRegressor` class was developed using NumPy.

The implementation includes:

- Initialization of coefficients and intercept.
- Prediction using matrix multiplication.
- Calculation of gradients for coefficients and intercept.
- Iterative parameter updates using gradient descent.
- Prediction on unseen test data.

## Model Comparison

The custom Gradient Descent implementation was compared against Scikit-learn's `LinearRegression` model using the R² score.

| Model | R² Score |
|---|---:|
| Scikit-learn Linear Regression | 0.43993 |
| Custom Gradient Descent | 0.43932 |

Both implementations achieved very similar R² scores on the same test dataset.

## Technologies Used

- Python
- NumPy
- Pandas / Scikit-learn
- Matplotlib

## Repository Contents

- `gradient-descent.ipynb` – Complete implementation, model training, and comparison.

## Key Learning Outcomes

- Understanding gradient descent in multiple linear regression.
- Implementing vectorized gradient calculations using NumPy.
- Understanding coefficient and intercept optimization.
- Comparing a custom machine learning implementation with Scikit-learn.

---

**Note:** This project is part of my journey to understand and implement machine learning algorithms from scratch.
