# Initial Project Plan

> This document records the initial project plan developed before the final assignment scope was refined. The final submitted implementation focuses only on the clean synthetic baseline experiment. The broader experiments are retained here as part of the project development history.

## From Loss to Learning: A From-Scratch Study of Linear Regression under Outliers

*31005 Machine Learning - Assignment 2 - Option 1*

## 1. Project Positioning and Aim

This project follows Assignment 2 Option 1 and studies a fundamental machine learning model in depth. The core aim is to implement a complete Linear Regression learning pipeline from scratch and use it to understand the connection between hypothesis, prediction, loss, gradient, optimization, and training.

The central learning pipeline is:

**Hypothesis → Prediction → Loss → Gradient → Optimizer → Training**

MSE, MAE, and Huber Loss will be implemented as three alternative loss functions within the same model and training framework. Controlled outliers will then be introduced to investigate how the choice of loss function changes gradients, parameter updates, and ultimately the behaviour of the learned model.

The project is designed around understanding mechanisms rather than only comparing final performance. The main analysis will therefore follow the causal chain:

**Outlier → Residual → Loss → Gradient → Parameter Update → Learned Model**

## 2. Main Research Question

How do MSE, MAE, and Huber Loss affect the gradient-based learning behaviour of a from-scratch Linear Regression model under outliers?

Supporting objectives are to: (1) establish clean-data behaviour; (2) study the effect of outlier magnitude and proportion; (3) compare vertical outliers and high-leverage points; (4) examine the effect of the Huber threshold; and (5) test whether the main behaviours observed under controlled synthetic conditions can also be seen in a small real-world example.

## 3. From-Scratch Machine Learning Implementation

The core model will be one-dimensional Linear Regression:

$$
\hat{y} = wx + b
$$

The hypothesis space is the set of linear functions parameterised by $w$ and $b$. Prediction, the three loss functions, analytical gradients, Batch Gradient Descent, and the complete training loop will be implemented directly. NumPy, Pandas, and Matplotlib may be used for numerical operations, data handling, and visualisation, but packaged Linear Regression models, packaged loss implementations, or automatic differentiation will not be used as the core implementation.

The project will derive the parameter gradients through the chain rule. Since $\hat{y} = wx + b$:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}}x
\qquad
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial \hat{y}}
$$

Parameters will be updated using Batch Gradient Descent:

$$
w^{(t+1)} = w^{(t)} - \eta\frac{\partial L}{\partial w}
\qquad
b^{(t+1)} = b^{(t)} - \eta\frac{\partial L}{\partial b}
$$

## 4. Loss Functions

Three loss functions will be implemented and analysed under the same model, data, and optimization framework:

- **Mean Squared Error (MSE):** quadratic penalty. Large residuals receive increasingly large penalties and gradient influence.
- **Mean Absolute Error (MAE):** linear penalty. The prediction-level gradient magnitude is bounded away from the non-differentiable point, which is expected to reduce sensitivity to large vertical residuals.
- **Huber Loss:** quadratic for small residuals and linear for large residuals. The threshold $\delta$ controls the transition and provides a tunable compromise between MSE-like and MAE-like behaviour.

The project will connect each mathematical definition to its corresponding code and to the observed training behaviour.

## 5. Primary Dataset: Controlled Synthetic Data

Controlled synthetic data will be the primary dataset because the project requires known ground truth and precise control over outlier interventions. A simple underlying relationship will be generated, for example:

$$
y = 3x + 2 + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

The true parameters are therefore known ($w_{\text{true}} = 3$ and $b_{\text{true}} = 2$). This allows the project to evaluate not only prediction error but also parameter recovery. The same clean data-generating process can then be modified in a controlled way by changing outlier magnitude, contamination proportion, or outlier type.

A clean test set generated from the original relationship will be retained for the main synthetic experiments. This makes it possible to test whether a model trained on contaminated data still recovers the underlying relationship and generalises to normal unseen samples.

## 6. Experimental Plan

| Experiment | Controlled Variable | Main Purpose | Key Evidence |
| --- | --- | --- | --- |
| 1. Clean Baseline | Loss function | Verify implementation and establish normal behaviour | Convergence, $w$/$b$, test error, fitted line |
| 2. Outlier Magnitude | Vertical-outlier magnitude | Trace how increasingly extreme residuals affect learning | Sample loss, gradient contribution, parameter shift |
| 3. Outlier Proportion | Contamination ratio | Study degradation as abnormal samples become more common | Parameter recovery, clean-test RMSE/MAE |
| 4. Outlier Type | Vertical vs high-leverage | Test whether different abnormal points act through different mechanisms | Gradient influence, $w$/$b$, fitted line |
| 5. Huber Threshold | $\delta$ | Study the transition between MSE-like and MAE-like behaviour | Gradient, parameter recovery, clean-test performance |
| 6. Real-World Validation | Small real $x \rightarrow y$ dataset | Check whether key synthetic observations also appear in a less controlled setting | MSE/MAE/Huber fitted lines, RMSE/MAE, unusual observations |

## 7. Evaluation and Analysis

The project will not use training loss alone as the final comparison because the models optimize different loss definitions. Evaluation will combine:

- **Prediction performance:** RMSE and MAE on a clean test set.
- **Parameter recovery:** $|\hat{w} - w_{\text{true}}|$ and $|\hat{b} - b_{\text{true}}|$ for synthetic experiments.
- **Learning behaviour:** loss curves, gradient magnitude, parameter trajectories, and changes in the fitted regression line.
- **Sample-level influence:** gradient contribution of selected outliers in key experiments.

The main interpretation will explain why a result occurs by tracing the effect from residual to loss, gradient, parameter update, and final model. Results will not be reduced to a simple ranking of which loss function performs best.

## 8. Verification and Reproducibility

Analytical gradients will be checked using finite-difference numerical gradient checking. After the from-scratch implementation is complete, a mature library implementation may be used only as an external sanity check, not as the project implementation.

The complete project will be developed in a self-contained Google Colab notebook. Random seeds, data-generation settings, initialization, train/test splits, and experimental conditions will be recorded. Before submission, the notebook will be rerun from a clean runtime to verify reproducibility.

## 9. Planned A2 Deliverables

The final A2 submission will contain the required public implementation link, project report, and implementation log.

- **Public implementation:** a self-contained Google Colab notebook containing the from-scratch implementation, experiments, evaluation, and visualisations.
- **Project report:** Task Definition, Research Question, Machine Learning Approach, Theory-to-Code explanation, Experimental Design, Results, Discussion, Limitations, and Potential Future Work.
- **Implementation log:** a chronological record of development, technical problems and solutions, AI-tool usage, verification, and remaining knowledge gaps together with attempts to resolve them.

## 10. Scope Boundary

The project will remain centred on Simple Linear Regression and loss-function behaviour. The real-world dataset will be used only as supporting validation and will not replace the controlled synthetic experiments. The project will not expand into a complex multivariate prediction system unless later feedback specifically requires it.

This keeps the scope manageable while preserving the technical depth of the main study: a transparent from-scratch learning pipeline, mathematically derived gradients, controlled outlier experiments, and evidence-based interpretation of model behaviour.
