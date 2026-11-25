#' autolearnR: Machine Learning for Biological Data
#'
#' autolearnR provides tools for supervised machine learning workflows,
#' with a focus on classification tasks in biological datasets.
#' The package supports dataset diagnostics, model training,
#' performance evaluation, model comparison, and prediction
#' in independent datasets.
#'
#' ## Main Workflow
#'
#' The package provides four complementary steps:
#'
#' - `learn_design()` - Evaluates dataset characteristics, identifies
#'   potential issues, and provides heuristic model recommendations.
#' - `learn_*()` - Trains classification models using Logistic Regression,
#'   Elastic Net, Decision Trees, Random Forest, or Support Vector Machines.
#' - `learn_compare()` - Combines and compares the performance of
#'   multiple trained models.
#' - `learn_predict()` - Applies frozen models to independent datasets
#'   to evaluate predictive generalization.
#'
#' ## Available Algorithms
#'
#' | Function | Algorithm |
#' |----------|-----------|
#' | `learn_logistic()` | Logistic Regression |
#' | `learn_elastic_net()` | Elastic Net |
#' | `learn_tree()` | Decision Tree |
#' | `learn_random_forest()` | Random Forest |
#' | `learn_svm()` | Support Vector Machine |
#'
#' ## Important Notes
#'
#' - The current implementation supports supervised classification.
#' - Missing values are diagnosed but are not automatically removed
#'   or imputed during model training.
#' - Model recommendations are heuristic and should not replace
#'   statistical judgment or independent validation.
#'
#' For installation instructions, examples, and further details,
#' visit the package repository:
#' [GitHub](https://github.com/Luiz-Garcia-R/autolearnR).
#'
#' @examples
#' \dontrun{
#' # Explore dataset characteristics
#' design <- learn_design(
#'   data = my_data,
#'   target_var = "Group",
#'   sample_col = "Sample"
#' )
#'
#' # Train a classification model
#' result <- learn_random_forest(
#'   data = my_data,
#'   target_var = "Group",
#'   sample_col = "Sample"
#' )
#'
#' # Apply the frozen model to an independent dataset
#' external_result <- learn_predict(
#'   model = result$frozen_model,
#'   new_data = validation_data
#' )
#' }
#'
#' @name autolearnR
#' @docType package
#' @keywords package
"_PACKAGE"
