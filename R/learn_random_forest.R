# =============================================================================
# LEARN RANDOM FOREST
# =============================================================================

#' Train and evaluate a Random Forest classification model
#'
#' Trains a Random Forest classifier using a stratified training/test split
#' and evaluates its predictive performance on held-out samples.
#'
#' The function supports binary and multiclass classification, calculates
#' standard classification metrics, and provides additional Random Forest
#' diagnostics, including out-of-bag (OOB) error and feature importance.
#'
#' For binary classification, ROC analysis and AUC calculation are also
#' performed. An optional permutation test can be used to evaluate whether
#' the model's predictive accuracy exceeds what would be expected by chance.
#'
#' @param data A data frame containing the predictor variables and the
#'   classification outcome.
#'
#' @param target_var Character string specifying the name of the target
#'   variable.
#'
#' @param feature_subset Optional character vector specifying the predictor
#'   variables to be used for model training. If \code{NULL}, all eligible
#'   predictors are considered.
#'
#' @param sample_col Character string specifying the column containing sample
#'   identifiers. Defaults to \code{"Sample"}.
#'
#' @param train_fraction Numeric value between 0 and 1 specifying the
#'   proportion of samples allocated to the training set. The split is
#'   stratified according to the target variable. Defaults to \code{0.7}.
#'
#' @param ntree Integer specifying the number of trees to grow. Defaults to
#'   \code{500}.
#'
#' @param mtry Number of predictors randomly sampled as candidates at each
#'   split. If \code{NULL} (default), the default value from
#'   \code{\link[randomForest]{randomForest}} is used, calculated as
#'   \code{floor(sqrt(ncol(x_train)))}.
#'
#' @param seed Integer specifying the random seed for reproducibility.
#'   Defaults to \code{123}.
#'
#' @param positive_class Optional character string specifying the positive
#'   class for binary classification. If \code{NULL}, the positive class is
#'   determined by the classification helper. Not applicable to multiclass
#'   classification.
#'
#' @param permutation_test Logical value indicating whether to perform a
#'   permutation test to evaluate model accuracy. Defaults to \code{FALSE}.
#'
#' @param nperm Integer specifying the number of permutations used in the
#'   permutation test. Defaults to \code{1000}.
#'
#' @param plot Logical value indicating whether to generate diagnostic plots.
#'   Defaults to \code{TRUE}.
#'
#' @details
#' The function performs the following steps:
#' \enumerate{
#'   \item Prepares the classification dataset and validates the predictors.
#'   \item Splits the data into training and test sets using stratified
#'   sampling.
#'   \item Trains a Random Forest classifier using
#'   \code{randomForest::randomForest()}.
#'   \item Predicts class labels and class probabilities for the test set.
#'   \item Calculates classification metrics and confusion matrix.
#'   \item Performs ROC analysis and calculates AUC for binary classification.
#'   \item Extracts feature importance and OOB diagnostics.
#'   \item Optionally performs a permutation test.
#'   \item Generates individual diagnostic plots and a combined plot when
#'   \code{plot = TRUE}.
#' }
#'
#' The Random Forest model uses bootstrap aggregation and random feature
#' selection to construct an ensemble of decision trees. The OOB error
#' provides an internal estimate of prediction error based on samples not
#' included in the bootstrap sample used to grow each tree.
#'
#' Feature importance is extracted from the trained model. The specific
#' importance measure is stored in the output object, together with the
#' corresponding feature importance values.
#'
#' When \code{permutation_test = TRUE}, the target labels are permuted
#' repeatedly and the model accuracy is evaluated under the permuted data.
#' The resulting empirical p-value provides an assessment of whether the
#' observed predictive performance exceeds that expected under the
#' permutation procedure.
#'
#' Classification performance metrics are accompanied by confidence
#' intervals estimated using non-parametric bootstrap resampling of the
#' held-out test samples. For binary classification, the AUC confidence
#' interval is estimated using the DeLong method.
#'
#' The returned object follows the common output structure used by
#' \code{autolearnR} classification algorithms. Algorithm-specific results
#' are stored in the \code{details} component, while shared evaluation
#' results are made available through standardized components.
#'
#' @return An object of class \code{"learn_random_forest"}, containing:
#' \describe{
#'   \item{call}{The original function call.}
#'   \item{algorithm}{Character string identifying the algorithm:
#'   \code{"random_forest"}.}
#'   \item{model_family}{Classification type, either \code{"binomial"} or
#'   \code{"multinomial"}.}
#'   \item{model}{The fitted Random Forest model returned by
#'   \code{randomForest::randomForest()}.}
#'   \item{frozen_model}{A model bundle containing the fitted model and
#'   associated metadata, including predictor names, factor levels, target
#'   levels, and relevant model parameters.}
#'   \item{features}{Character vector containing the predictor variables
#'   used for model training.}
#'   \item{train_samples}{Sample identifiers assigned to the training set.}
#'   \item{test_samples}{Sample identifiers assigned to the test set.}
#'   \item{prediction_data}{Data frame containing sample identifiers,
#'   observed classes (\code{Observed}), and predicted classes
#'   (\code{Predicted}) for the test set.}
#'   \item{predictions}{Predicted class labels for the test set.}
#'   \item{probabilities}{Predicted class probabilities for the test set.}
#'   \item{confusion_matrix}{Confusion matrix summarizing observed and
#'   predicted classifications.}
#'   \item{confusion_table}{Tabular representation of the confusion matrix.}
#'   \item{metrics}{Named list of classification performance metrics,
#'   including accuracy, sensitivity, specificity, precision, F1-score,
#'   and AUC when applicable.}
#'   \item{roc}{ROC object for binary classification. Returns
#'   \code{NULL} for multiclass classification.}
#'   \item{confidence_intervals}{A list containing confidence intervals for the
#'   classification performance metrics, estimated using non-parametric
#'   bootstrap resampling.}
#'   \item{details}{A list containing Random Forest-specific results:
#'   \describe{
#'     \item{importance}{Data frame containing feature importance values.}
#'     \item{top_features}{Character vector containing the top-ranked
#'     predictors according to feature importance.}
#'     \item{importance_metric}{Name of the importance measure used.}
#'     \item{oob_error}{Final out-of-bag error estimate.}
#'     \item{oob_trajectory}{Data frame containing the OOB error trajectory
#'     across the number of trees.}
#'     \item{permutation}{Results of the optional permutation test,
#'     including observed accuracy and empirical p-value. Returns
#'     \code{NULL} when the test is not performed.}
#'   }}
#'   \item{plots}{A list containing individual diagnostic plots, including
#'   confusion matrix, OOB error, feature importance, ROC curve, and
#'   permutation test plots when applicable. Empty when \code{plot = FALSE}.}
#'   \item{combined_plot}{Combined diagnostic plot. Returns \code{NULL}
#'   when \code{plot = FALSE}.}
#'   \item{parameters}{A list containing the parameters used during model
#'   training and evaluation.}
#' }
#'
#' @examples
#' # Example: Train a Random Forest algorithm using the iris dataset
#' # to discriminate between versicolor, virginica and setosa species.
#'
#' # Select species
#' iris_data <- iris[
#'   iris$Species %in% c("versicolor", "virginica", "setosa"),
#' ]
#'
#' # Create columns "Sample" and "Species"
#' iris_data$Sample <- rownames(iris_data)
#' iris_data$Species <- droplevels(iris_data$Species)
#'
#' # Train and test model
#' rf_model <- learn_random_forest(
#'   data = iris_data,
#'   target_var = "Species",
#'   feature_subset = c(
#'     "Sepal.Length",
#'     "Sepal.Width",
#'     "Petal.Length",
#'     "Petal.Width"
#'   ),
#'   sample_col = "Sample",
#'   train_fraction = 0.7,
#'   seed = 123,
#'   plot = TRUE
#' )
#'
#' @seealso
#' \code{\link[randomForest]{randomForest}},
#' \code{\link[randomForest]{importance}},
#' \code{\link[pROC]{roc}},
#' \code{\link{learn_elastic_net}}
#'
#' @export

learn_random_forest <- function(
    data,
    target_var,
    feature_subset = NULL,
    sample_col = "Sample",
    train_fraction = 0.7,
    ntree = 500,
    mtry = NULL,
    seed = 123,
    positive_class = NULL,
    permutation_test = FALSE,
    nperm = 1000,
    plot = TRUE
) {

# =============================================================================
# 0. basic checks
# =============================================================================

  .check_dependencies(c(
  "randomForest",
  "pROC",
  "ggplot2",
  "patchwork",
  "caret"
))

# =============================================================================
# 1. Prepare classification data
# =============================================================================

prepared <- .prepare_classification_data(
  data = data,
  target_var = target_var,
  feature_subset = feature_subset,
  sample_col = sample_col
)

X <- prepared$x
y <- prepared$y

# =============================================================================
# 2. Predictor validation
# =============================================================================

valid_predictors <- vapply(
  X,
  function(x) {
    is.numeric(x) || is.factor(x)
  },
  logical(1)
)

invalid_predictors <- !valid_predictors

if (any(invalid_predictors)) {

  stop(
    "Unsupported predictor types detected: ",
    paste(
      colnames(X)[invalid_predictors],
      collapse = ", "
    ),
    ". Only numeric and factor predictors are supported."
  )
}

# Validate factor levels
factor_features <- names(X)[
  vapply(X, is.factor, logical(1))
]

for (feature in factor_features) {

  if (nlevels(X[[feature]]) > 53) {

    stop(
      "Factor variable '", feature,
      "' exceeds the maximum of 53 levels supported by randomForest."
    )
  }
}

# Identify predictor types
predictor_types <- .get_predictor_types(X)

# =============================================================================
# 3. Train / Test split
# =============================================================================

set.seed(seed)

split_idx <- .make_stratified_split(
  y = y,
  p = train_fraction
)

train_idx <- split_idx$train
test_idx  <- split_idx$test

X_train <- X[
  train_idx,
  ,
  drop = FALSE
]

X_test <- X[
  test_idx,
  ,
  drop = FALSE
]

y_train <- y[train_idx]
y_test  <- y[test_idx]

# =============================================================================
# 4. Train Random Forest
# =============================================================================

# Train null model
rf_args <- list(
  x = X_train,
  y = y_train,
  ntree = ntree,
  importance = TRUE
)

if (!is.null(mtry)) {
  rf_args$mtry <- mtry
}

rf_model <- do.call(
  randomForest::randomForest,
  rf_args
)

# =============================================================================
# 5. Validate levels and model family
# =============================================================================

# Store categorical predictor levels
factor_levels <- lapply(
  X_train[
    vapply(X_train, is.factor, logical(1))
  ],
  levels
)

# Identify model family
model_family <- if (nlevels(y_train) == 2) {
  "binomial"
} else {
  "multinomial"
}

# Define positive class
if (model_family == "binomial") {

  if (is.null(positive_class)) {
    positive_class <- levels(y_train)[2]
  }

  if (
    !is.character(positive_class) ||
    length(positive_class) != 1 ||
    is.na(positive_class) ||
    !positive_class %in% levels(y_train)
  ) {
    stop(
      "'positive_class' must match one of the target classes."
    )
  }

} else {

  if (!is.null(positive_class)) {
    stop(
      "'positive_class' is only applicable to binary classification."
    )
  }

}

# =============================================================================
# 6. Predictions
# =============================================================================

pred_class <- predict(
  rf_model,
  newdata = X_test,
  type = "response"
)

pred_prob <- predict(
  rf_model,
  newdata = X_test,
  type = "prob"
)

# =============================================================================
# 7. Classification metrics
# =============================================================================

performance <- .classification_metrics(
  predicted = pred_class,
  observed = y_test,
  n_boot = 2000,
  conf_level = 0.95,
  seed = 123 + 1
)

cm <- performance$confusion_matrix
cm_tab <- performance$confusion_table
metrics <- performance$metrics
accuracy <- metrics$accuracy
confidence_intervals <- performance$confidence_intervals

# =============================================================================
# 8. ROC / AUC
# =============================================================================

roc_obj <- NULL
auc_val <- NA_real_
auc_ci <- NULL

if (model_family == "binomial") {

  roc_results <- .classification_roc(
    observed = y_test,
    probabilities = pred_prob,
    positive_class = positive_class
  )

  roc_obj <- roc_results$roc
  auc_val <- roc_results$auc

  positive_class <- roc_results$positive_class

auc_ci_raw <- roc_results$auc_ci

auc_ci <- list(
  lower = unname(auc_ci_raw[1]),
  estimate = unname(auc_ci_raw[2]),
  upper = unname(auc_ci_raw[3]),
  method = "delong"
)

# Update confidence_interval
if (!is.null(auc_ci)) {

  auc_ci_row <- data.frame(
    Metric = "auc",
    Class = NA_character_,
    Estimate = auc_ci$estimate,
    Lower = auc_ci$lower,
    Upper = auc_ci$upper,
    N_valid = NA_integer_,
    N_boot = NA_integer_,
    Method = "delong"
  )

  confidence_intervals <- rbind(
    confidence_intervals,
    auc_ci_row
  )

  }
}

metrics$auc <- auc_val
metrics$auc_ci <- auc_ci

# =============================================================================
# 9. Build frozen model
# =============================================================================

frozen_model <- list(

  # Schema information
  schema_version = 1L,

  # Identification
  algorithm = "random_forest",
  model = rf_model,
  model_family = model_family,

  # Predictor information
  predictor_names = colnames(X_train),
  predictor_types = predictor_types,

  # Training data
  training_data = list(
    predictors = X_train,
    response = y_train
  ),

  preprocessing = list(
    method = "native_randomForest",
    factor_levels = factor_levels

  ),

  # Response information
  response = list(
    target_var = target_var,
    target_levels = levels(y_train),

    positive_class = if (model_family == "binomial") {
      positive_class
    } else {
      NULL
    },

    negative_class = if (model_family == "binomial") {
      setdiff(levels(y_train), positive_class)
    } else {
      NULL
    }
  ),

  # Prediction settings
  prediction = list(
    method = "maximum_probability",
    threshold = NULL
  ),

  # Algorithm-specific parameters
  algorithm_parameters = list(
    ntree = ntree,
    mtry = rf_model$mtry,
    importance = TRUE
  ),

  # Reproducibility
  reproducibility = list(
    seed = seed
  ),

  # Original evaluation reference
  reference = list(
    evaluation_method = "holdout",
    n_samples = length(y_test),
    sample_ids = rownames(X_test),

    observed = factor(
      y_test,
      levels = levels(y)
    ),

    predictions = factor(
      pred_class,
      levels = levels(y)
    ),

    probabilities = pred_prob,
    metrics = metrics,
    confidence_intervals = confidence_intervals,
    confusion_matrix = cm,
    confusion_table = cm_tab,
    roc = roc_obj,
    auc = auc_val,
    auc_ci = auc_ci,
    confidence_intervals = confidence_intervals,
    evaluation_parameters = list(
      train_fraction = train_fraction,
      n_boot = 2000,
      conf_level = 0.95,
      bootstrap_seed = seed + 1
    )
  )

)

class(frozen_model) <- c(
  "autolearn_random_forest",
  "list"
)

# =============================================================================
# 10. Prediction data
# =============================================================================

prediction_data <- data.frame(
  Sample = rownames(X_test),

  Observed = factor(
    y_test,
    levels = levels(y)
  ),

  Predicted = factor(
    pred_class,
    levels = levels(y)
  ),

  stringsAsFactors = FALSE
)

# =============================================================================
# 11. Feature importance
# =============================================================================

importance_results <- .rf_feature_importance(
  model = rf_model,
  top_n = 10
)

imp_df <- importance_results$importance
top_features <- importance_results$top_features
importance_col <- importance_results$importance_metric

# =============================================================================
# 12. OOB diagnostics
# =============================================================================

oob_results <- .rf_oob_diagnostics(
  model = rf_model
)

oob_error <- oob_results$oob_error
oob_df <- oob_results$trajectory

# =============================================================================
# 13. Permutation test
# =============================================================================

perm_acc <- NULL
perm_p <- NULL
perm_results <- NULL

if (permutation_test) {

  perm_results <- .rf_permutation_test(
    x_train = X_train,
    y_train = y_train,
    x_test = X_test,
    y_test = y_test,
    observed_accuracy = accuracy,
    ntree = ntree,
    mtry = rf_model$mtry,
    nperm = nperm,
    seed = seed
  )

  perm_acc <- perm_results$accuracy
  perm_p <- perm_results$p_value
}

# =============================================================================
# 14. Plots
# =============================================================================

plots <- list()
gg_roc <- NULL
combined_plot <- NULL

if (plot) {

  # ===============================================
  # 14.1. Confusion matrix
  # ===============================================

  gg_cm <- .plot_confusion_matrix(
    confusion_table = cm_tab
  )

  plots$confusion <- gg_cm

  # ===============================================
  # 14.2. OOB error
  # ===============================================

  gg_oob <- .plot_oob_error(
    oob_trajectory = oob_df
  )

  plots$oob <- gg_oob

  # ===============================================
  # 14.3. Feature importance
  # ===============================================

  gg_imp <- .plot_feature_importance(
    feature_importance = top_features,
    importance_col = importance_col
  )

  plots$importance <- gg_imp

  # ===============================================
  # 14.4. ROC
  # ===============================================

  if (model_family == "binomial" && !is.null(roc_obj)) {

    gg_roc <- .plot_roc_curve(
      roc_object = roc_obj,
      auc_value = auc_val
    )

    plots$roc <- gg_roc

  }

  # ===============================================
  # 14.5. Combined plot
  # ===============================================

  combined_plot <- .combine_classification_plots(
    plots = list(
      confusion = gg_cm,
      oob = gg_oob,
      roc = gg_roc,
      importance = gg_imp
    )
  )

  # ===============================================
  # 14.6. Permutation plot
  # ===============================================

  if (!is.null(perm_acc)) {

    gg_perm <- .plot_permutation_test(
      permutation_accuracy = perm_acc,
      observed_accuracy = accuracy,
      p_value = perm_p
    )

    plots$permutation <- gg_perm

  }

  # ===============================================
  # 14.7. Print plots
  # ===============================================

  if (!is.null(plots$permutation)) {
    print(plots$permutation)
  }

  if(!is.null(combined_plot)) {
    print(combined_plot)
  }

}

# =============================================================================
# 15. Output
# =============================================================================

obj <- list(

  # Identification
  call = match.call(),
  algorithm = "random_forest",
  model_family = model_family,

  # Fitted models
  model = rf_model,
  frozen_model = frozen_model,

  # Data information
  features = colnames(X),
  train_samples = rownames(X_train),
  test_samples = rownames(X_test),

  # Predictions
  prediction_data = prediction_data,
  predictions = pred_class,
  probabilities = pred_prob,

  # Performance evaluation
  confusion_matrix = cm,
  confusion_table = cm_tab,
  metrics = metrics,
  roc = roc_obj,
  auc_ci = auc_ci,
  confidence_intervals = confidence_intervals,

  # Algorithm-specific results
  details = list(
    importance = imp_df,
    top_features = top_features,
    importance_metric = importance_col,
    oob_error = oob_error,
    oob_trajectory = oob_df,
    permutation = if (permutation_test) {
      list(
        accuracy = perm_acc,
        p_value = perm_p
      )
    } else {
      NULL
    }

  ),

  # Visualization
  plots = plots,
  combined_plot = combined_plot,

  # Parameters
  parameters = list(
    target_var = target_var,
    feature_subset = feature_subset,
    sample_col = sample_col,
    train_fraction = train_fraction,
    ntree = ntree,
    mtry = rf_model$mtry,
    seed = seed,
    positive_class = positive_class,
    permutation_test = permutation_test,
    nperm = nperm,
    plot = plot
  )
)

class(obj) <- "learn_random_forest"

print(obj)

return(invisible(obj))

}

# =============================================================================
# Print S3
# =============================================================================

#' Print method for learn_random_forest objects
#'
#' Displays a formatted summary of a \code{learn_random_forest} object.
#'
#' @param x An object of class \code{learn_random_forest}.
#' @param ... Further arguments (ignored).
#' @return Invisibly returns \code{x}.
#' @method print learn_random_forest
#' @export

print.learn_random_forest <- function(x, ...) {

  # Get confidence intervals
  get_ci <- function(metric) {
    .get_ci(
      confidence_intervals = x$confidence_intervals,
      metric = metric
    )
  }

  .print_header("Random Forest Results")

  # General information
  n_train <- length(x$train_samples)
  n_test  <- length(x$test_samples)
  n_feat  <- length(x$features)

  classes <- paste(
    unique(x$prediction_data$Observed),
    collapse = " vs "
  )

  # ===============================================
  # Data and setup
  # ===============================================

  .print_block("Data & Setup", function() {

    cat(
      "Target Variable:        ",
      x$parameters$target_var,
      " (", classes, ")\n",
      sep = ""
    )

    cat(
      "Model Family:          ",
      x$model_family,
      "\n"
    )

    cat(
      "Samples (Train/Test):  ",
      n_train,
      "/",
      n_test,
      "\n"
    )

    cat(
      "Features Available:    ",
      n_feat,
      "\n"
    )

    cat(
      "Number of Trees:       ",
      x$parameters$ntree,
      "\n"
    )

    cat(
      "mtry:                  ",
      x$parameters$mtry,
      "\n"
    )

  })

  # ===============================================
  # Performance
  # ===============================================

  .print_block("Performance (Test Set; estimates with 95% CI)", function() {

    cat(
      "OOB Error Rate:        ",
      round(x$details$oob_error, 4),
      "\n"
    )

    cat(
      "Accuracy:              ",
      .format_metric_ci(
        x$metrics$accuracy,
        get_ci("accuracy")
      ),
      "\n"
    )

    cat(
      "Kappa:                 ",
      .format_metric_ci(
        x$metrics$kappa,
        get_ci("kappa")
      ),
      "\n"
    )

    cat(
      "Macro F1-Score:        ",
      .format_metric_ci(
        x$metrics$macro_f1,
        get_ci("macro_f1")
      ),
      "\n"
    )

    if (
      !is.null(x$metrics$auc) &&
      is.finite(x$metrics$auc)
    ) {

      cat(
        "AUC (ROC):             ",
        .format_metric_ci(
          x$metrics$auc,
          if (!is.null(x$auc_ci)) {
            c(x$auc_ci$lower, x$auc_ci$upper)
          } else {
            NULL
          }
        ),
        "\n"
      )

    }

    if (!is.null(x$details$permutation)) {

      cat(
        "Permutation p-value:   ",
        round(x$details$permutation$p_value, 4),
        "\n"
      )
    }
  }
)

  invisible(x)

}

