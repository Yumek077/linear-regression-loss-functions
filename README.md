# Linear Regression with Different Loss Functions

## Project Overview

This project implements one-dimensional linear regression from scratch using NumPy. It examines how mean squared error (MSE), mean absolute error (MAE), and Huber loss affect analytical gradients, gradient-descent updates, and the final fitted models.

## Final Implementation

The [`final/`](final/) directory contains the focused implementation used for the final assignment. Its main pipeline is:

**Residual → Loss → Analytical Gradient → Parameter Update → Final Model**

The notebook verifies the analytical gradients using central finite differences and evaluates the three loss functions on a clean synthetic baseline with Gaussian noise.

## Exploratory Development

The [`exploratory/`](exploratory/) directory contains an earlier, broader development notebook with additional experiments. These experiments are retained as a record of project development and exploration; they are not all part of the final submitted study.

## Repository Structure

```text
linear-regression-loss-functions/
├── README.md
├── final/
│   └── Linear_Regression_Loss_Functions_Final.ipynb
└── exploratory/
    └── Linear_Regression_Full_Experiments.ipynb
```

- `final/`: Clean synthetic baseline and final assignment implementation.
- `exploratory/`: Earlier development work and Experiments 1–5.

## Tools

- Python
- NumPy
- Matplotlib
- Jupyter / Google Colab
