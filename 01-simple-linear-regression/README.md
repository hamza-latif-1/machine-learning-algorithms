# Simple Linear Regression from Scratch

## Overview

This project implements Simple Linear Regression from scratch using Python and NumPy.

The model uses the Ordinary Least Squares (OLS) method to calculate the optimal slope and intercept.

## Mathematical Formulation

The prediction equation is:

$$
\hat{y} = mx + b
$$

Where:
- `m` = Slope
- `b` = Intercept
- `x` = Input feature
- `ŷ` = Predicted output

### Slope

$$
m = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^{n}(x_i-\bar{x})^2}
$$

### Intercept

$$
b = \bar{y} - m\bar{x}
$$

## Technologies Used

- Python
- NumPy

## Implementation

The implementation contains a `MyLinearRegression` class with two main methods:

- `fit()` – Calculates the slope and intercept from training data.
- `predict()` – Generates predictions using the learned parameters.

## How to Run

Install NumPy:

```bash
pip install numpy
```

Run the implementation:

```bash
python main.py
```

## Learning Objectives

- Understand the mathematics behind Simple Linear Regression.
- Implement OLS using Python and NumPy.
- Understand how a regression model learns its parameters.
