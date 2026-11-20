# Stochastic Gradient Descent for Multiple Linear Regression

This project implements **Stochastic Gradient Descent (SGD) from scratch** using NumPy and compares its performance with Scikit-learn's `LinearRegression` model on the Diabetes dataset.

## Project Overview

The notebook demonstrates how to:

- Load and inspect a dataset using Scikit-learn.
- Split the data into training and testing sets.
- Train a multiple linear regression model with Scikit-learn.
- Implement a simple stochastic gradient descent regressor from scratch.
- Generate predictions and compare the models using the coefficient of determination (R² score).

## Dataset

The project uses Scikit-learn's built-in **Diabetes dataset** through `sklearn.datasets.load_diabetes`.

- **Total observations:** 442
- **Features:** 10
- **Target:** A quantitative measure of disease progression
- **Training set:** 353 observations
- **Test set:** 89 observations
- **Train/test split:** 80% training and 20% testing
- **Random state:** 2

## Implementation

### 1. Scikit-learn Linear Regression

First, a baseline model is trained using Scikit-learn's `LinearRegression`:

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
```

The fitted model is used to predict the target values for the test set.

### 2. Stochastic Gradient Descent from Scratch

A `StochasticRegressor` class is implemented using NumPy. It has two main methods:

- `fit(X_train, y_train)`: initializes the intercept and coefficients, then iteratively updates them using gradients calculated from individual randomly selected training observations.
- `predict(X_test)`: calculates predictions using the learned coefficients and intercept.

The parameter update is based on the squared-error loss for one selected observation. For a prediction:

\[
\hat{y}^{(i)} = \mathbf{x}^{(i)} \cdot \mathbf{w} + b
\]

The gradients used in the implementation are:

**Gradient with respect to the intercept:**

\[
\frac{\partial L_i}{\partial b} = -2(y^{(i)}-\hat{y}^{(i)})
\]

**Gradient with respect to the coefficients:**

\[
\frac{\partial L_i}{\partial \mathbf{w}} = -2(y^{(i)}-\hat{y}^{(i)})\mathbf{x}^{(i)}
\]

The parameters are updated using the learning rate:

\[
b \leftarrow b - \alpha\frac{\partial L_i}{\partial b}
\]

\[
\mathbf{w} \leftarrow \mathbf{w} - \alpha\frac{\partial L_i}{\partial \mathbf{w}}
\]

In the notebook, the SGD model is initialized and trained with:

```python
sgd = StochasticRegressor(learning_rate=0.02, epochs=35)
sgd.fit(X_train, y_train)
```

At each epoch, the implementation performs as many parameter updates as there are training observations. Each update selects one training row at random **with replacement** using `np.random.randint`.

## Model Evaluation and Results

Both models are evaluated on the same test set using the **R² score** from `sklearn.metrics.r2_score`.

| Model | Test R² score |
|---|---:|
| Scikit-learn Linear Regression | 0.43993 |
| Custom Stochastic Gradient Descent | 0.44358 |

In this particular run, the custom SGD model achieved a slightly higher test R² score. The results are close, showing that the from-scratch implementation reached comparable test-set performance in this experiment.

> **Note:** The SGD implementation uses random sampling, so its learned parameters and evaluation score may vary between runs. The train/test split is reproducible because `random_state=2` is specified, but the SGD sampling itself does not use a fixed random seed.

## Requirements

- Python 3
- NumPy
- Scikit-learn
- Jupyter Notebook (or JupyterLab)

Install the required packages with:

```bash
pip install numpy scikit-learn notebook
```

## Run the Notebook

1. Clone or download this repository.
2. Install the required packages.
3. Open the project notebook in Jupyter:

   ```bash
   jupyter notebook
   ```

4. Run `gradient-descent.ipynb` from the `03-gradient-descent/stochastic-gradient-descent/` directory.

## Project Structure

```text
03-gradient-descent/
└── stochastic-gradient-descent/
    ├── gradient-descent.ipynb
    └── README.md
```

## Learning Objective

The goal of this project is to understand the mechanics of stochastic gradient descent by implementing parameter updates manually and comparing the resulting regression model with Scikit-learn's implementation of ordinary least squares linear regression.
