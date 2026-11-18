# Gradient Descent from Scratch

## Overview

This project demonstrates the implementation of **Gradient Descent from scratch using Python and NumPy** to optimize the parameters of a Simple Linear Regression model.

Gradient Descent is an optimization algorithm that minimizes a cost function by iteratively updating model parameters in the direction opposite to the gradient.

The implementation and visualizations are provided in a Jupyter Notebook.

## 1. Mathematical Formulation

### Prediction Equation

For Simple Linear Regression, the predicted value is:

$$
\hat{y} = mx + b
$$

Where:

- \(m\) = Slope (coefficient)
- \(b\) = Intercept
- \(x\) = Input feature
- $\hat{y}$ = Predicted output

### Cost Function

To measure the error between actual and predicted values, we use the Sum of Squared Errors (SSE):

$$
J(m,b) = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Substituting the prediction equation:

$$
J(m,b) = \sum_{i=1}^{n}(y_i-(mx_i+b))^2
$$

The objective of Gradient Descent is to minimize this cost function.

### Partial Derivatives

We calculate the partial derivatives of the cost function with respect to the slope (\(m\)) and intercept (\(b\)).

**1. Derivative with respect to slope**

$$
\frac{\partial J}{\partial m} = -2 \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)
$$

**2. Derivative with respect to intercept**

$$
\frac{\partial J}{\partial b} = -2 \sum_{i=1}^{n} (y_i - \hat{y}_i)
$$

These derivatives represent how the cost changes with respect to each parameter.

### Parameter Updates

We update both parameters using the learning rate and their respective gradients.

**Slope update:**

$$
m = m - \alpha \frac{\partial J}{\partial m}
$$

**Intercept update:**

$$
b = b - \alpha \frac{\partial J}{\partial b}
$$

Both gradients are calculated using the current parameter values before updating either parameter.

This process is repeated for a specified number of epochs until the cost is minimized or convergence is achieved.

## 2. Implementation

The algorithm is implemented using:

- **Python** – Programming language
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Jupyter Notebook** – Implementation and experimentation

The implementation includes:

- Initializing model parameters.
- Calculating gradients.
- Updating slope and intercept.
- Repeating the optimization process over multiple epochs.
- Making predictions using the optimized parameters.

## 3. Visualization

The notebook visualizes how Gradient Descent updates the regression line over successive iterations.

It helps demonstrate how the model gradually adjusts its slope and intercept to find a better-fitting regression line.

## 4. Key Concepts Learned

- Intuition behind Gradient Descent.
- Mathematical formulation of the algorithm.
- Cost function and partial derivatives.
- Learning rate and epochs.
- Simultaneous parameter updates.
- Optimization of Simple Linear Regression parameters.
- Visualization of regression line updates.

## 5. Repository Structure

```text
03-gradient-descent/
│
├── gradient_descent.ipynb
└── README.md
```

## Objective

The objective of this project is to understand how Gradient Descent works mathematically and implement the optimization process independently rather than relying on a pre-built machine learning library.
