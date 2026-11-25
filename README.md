# autolearnR

<!-- badges: start --> <!-- badges: end -->

<table>
<tr>
<td width="70%" valign="top">

**Machine learning for biological data, from design to independent validation.**

autolearnR is an R package designed to streamline supervised machine learning workflows, with a focus on classification tasks.

It provides tools for dataset diagnostics, model training, performance evaluation, algorithm comparison, and prediction in independent datasets.

</td>

<td width="30%" align="center" valign="middle">

<img src="man/figures/logo_autolearnR.png" alt="autolearnR package logo" width="160">

</td>
</tr>
</table>

## Installation

You can install the development version directly from GitHub:

``` r
install.packages("remotes")
remotes::install_github("Luiz-Garcia-R/autolearnR")
```

## Getting started

The recommended workflow consists of four steps:

1.  **Design:** Explore dataset characteristics and obtain model recommendations using `learn_design()`.
2.  **Train:** Fit classification algorithms using the `learn_*()` functions.
3.  **Compare:** Evaluate and compare trained models using `learn_compare()`.
4.  **Validate:** Apply frozen models to independent datasets using `learn_predict()`.

## Input Data Format

autolearnR uses a single data frame containing the target variable and candidate predictors.

- Each row represents one sample.
- Each column represents a predictor or the target variable.
- An optional sample identifier column can be supplied through `sample_col`.
- The target variable must be a factor, character, or logical vector with at least two observed classes.

Example:

| Sample | Group   | Age | Biomarker_A | Biomarker_B |
|--------|---------|-----|-------------|-------------|
| S1     | Control | 35  | 2.5         | 10.2        |
| S2     | Control | 42  | 3.1         | 9.5         |
| S3     | Case    | 51  | 5.5         | 15.0        |
| S4     | Case    | 47  | 6.0         | 13.3        |

**Note:** Missing values are diagnosed but are not automatically removed or imputed during model training.

## Main Functions

### Dataset design

`learn_design()` – Evaluates dataset characteristics, identifies potential issues, and recommends classification algorithms.

### Classification algorithms

| Function                | Algorithm              |
|-------------------------|------------------------|
| `learn_logistic()`      | Logistic Regression    |
| `learn_elastic_net()`   | Elastic Net            |
| `learn_tree()`          | Decision Tree          |
| `learn_random_forest()` | Random Forest          |
| `learn_svm()`           | Support Vector Machine |

### Model evaluation

- `learn_compare()` – Combines and compares the performance of multiple trained models.
- `learn_predict()` – Applies frozen models to independent datasets and evaluates predictive generalization when reference labels are available.

## Minimal Example

``` r
library(autolearnR)

# Dataset diagnostics
design <- learn_design(
  data = learn_data,
  target_var = "Group",
  sample_col = "Sample"
)

# Train classification models
result_logistic <- learn_logistic(
  data = learn_data,
  target_var = "Group",
  sample_col = "Sample",
  seed = 123
)

result_elastic <- learn_elastic_net(
  data = learn_data,
  target_var = "Group",
  sample_col = "Sample",
  seed = 123
)

# Compare models
comparison <- learn_compare(
  results = list(
    logistic = result_logistic,
    elastic_net = result_elastic
  ),
  plot = TRUE
)

# Predict in an independent dataset
external_result <- learn_predict(
  model = result_logistic$frozen_model,
  new_data = validation_data
)
```

## Important Considerations

- **Classification only:** The current implementation supports supervised classification tasks.
- **Missing values:** Training functions require missing values to be handled before model fitting.
- **Model comparison:** Models should be evaluated using compatible test observations and class definitions.
- **Independent validation:** External evaluation provides evidence of generalization beyond the original test dataset.

## Contact

For questions, suggestions, bug reports, or contributions, open an issue or pull request on GitHub:

<https://github.com/Luiz-Garcia-R/autolearnR>

Thank you for using autolearnR!
