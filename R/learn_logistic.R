# =============================================================================
# LEARN LOGISTIC
# =============================================================================

#' Logistic Regression Classification
#'
#' Fits a binary logistic regression model using a binomial generalized linear
#' model (GLM) with a logit link function. The function performs stratified
#' train/test splitting, predictor encoding, model fitting, and independent
#' performance evaluation.
#'
#' Model coefficients, standard errors, z-statistics, p-values, odds ratios
#' (OR), and 95% Wald confidence intervals are provided for model
#' interpretation. Classification performance is evaluated using a confusion
#' matrix, Accuracy, Kappa, class-wise Precision, Recall, Specificity, and
#' F1-score, as well as macro-averaged metrics and ROC-AUC.
#'
#' Two diagnostic plots are available: a confusion matrix and a ROC curve.
#'
#' @param data A data frame containing the classification target and
#'   predictor variables.
#'
#' @param target_var Character string specifying the name of the target
#'   variable. The target must contain exactly two classes.
#'
#' @param feature_subset Optional character vector specifying the predictor
#'   variables to be included in the model. If \code{NULL}, all eligible
#'   variables except the target and sample identifier are used.
#'
#' @param sample_col Character string specifying the column containing sample
#'   identifiers. Defaults to \code{"Sample"}.
#'
#' @param train_fraction Numeric value between 0 and 1 specifying the
#'   proportion of samples allocated to the training set. Defaults to
#'   \code{0.7}.
#'
#' @param positive_class Optional character string specifying the positive
#'   target class. If \code{NULL}, the second factor level is used as the
#'   positive class.
#'
#' @param threshold Numeric classification threshold between 0 and 1.
#'   Samples with predicted probabilities equal to or greater than this
#'   threshold are assigned to the positive class. Defaults to \code{0.5}.
#'
#' @param seed Integer specifying the random seed for reproducible
#'   train/test splitting. Defaults to \code{123}.
#'
#' @param plot Logical value indicating whether diagnostic plots should be
#'   generated. Defaults to \code{TRUE}.
#'
#' @return An object of class \code{learn_logistic}, containing:
#' \describe{
#'   \item{model}{Fitted binomial GLM with a logit link function.}
#'   \item{frozen_model}{A list containing the fitted model, predictor
#'   encoding information, response levels, classification threshold,
#'   and settings required to reproduce the model predictions.}
#'   \item{prediction_data}{A data frame containing sample identifiers,
#'   observed classes, and predicted classes for the test set.}
#'   \item{predictions}{Predicted class labels for the test set.}
#'   \item{probabilities}{Predicted class probabilities for the test set.}
#'   \item{confusion_matrix}{Confusion matrix summarizing classification
#'   performance.}
#'   \item{confusion_table}{Confusion matrix in table format.}
#'   \item{metrics}{Classification performance metrics, including Accuracy,
#'   Kappa, class-wise Precision, Recall, Specificity, and F1-score,
#'   macro-averaged metrics, and ROC-AUC.}
#'   \item{roc}{ROC curve object generated using the test set predictions.}
#'   \item{confidence_intervals}{A list containing confidence intervals for the
#'   classification performance metrics, estimated using non-parametric
#'   bootstrap resampling.}
#'   \item{details}{A list containing algorithm-specific results:
#'     \describe{
#'       \item{coefficients}{Estimated predictor coefficients and associated
#'       statistics, including odds ratios and confidence intervals.}
#'       \item{all_coefficients}{Complete coefficient table, including
#'       the intercept.}
#'       \item{odds_ratios}{Odds ratios, 95% Wald confidence intervals,
#'       and corresponding p-values for the predictors.}
#'       \item{selected_features}{Predictors with estimable coefficients
#'       in the fitted model.}
#'       \item{feature_diagnostics}{Summary of predictor encoding and
#'       zero-variance filtering.}
#'       \item{removed_features}{Predictors removed due to zero variance.}
#'       \item{aliased_coefficients}{Names of coefficients that could not
#'       be estimated due to linear dependencies.}
#'       \item{converged}{Logical value indicating whether the model
#'       optimization algorithm converged.}
#'     }
#'   }
#'   \item{plots}{List containing the confusion matrix and ROC curve plots,
#'   when \code{plot = TRUE}.}
#'   \item{combined_plot}{Combined diagnostic plot, when
#'   \code{plot = TRUE}.}
#'   \item{parameters}{List containing the arguments and settings used
#'   during model fitting.}
#' }
#'
#' @details
#' The function fits a classical logistic regression model using
#' \code{\link[stats]{glm}} with \code{family = binomial(link = "logit")}.
#'
#' Categorical predictors are converted into factors and encoded using
#' dummy variables. Predictor levels and encoding structures are determined
#' from the training set and subsequently applied to the test set.
#' Predictors with zero variance in the training set are removed before
#' model fitting.
#'
#' The dataset is divided into training and test sets using stratified
#' sampling to preserve class proportions. Model fitting is performed
#' exclusively on the training set, while classification performance is
#' evaluated independently on the held-out test set.
#'
#' Odds ratios are calculated by exponentiating the estimated regression
#' coefficients. Their 95% confidence intervals are calculated using
#' Wald-based approximations.
#'
#' The \code{selected_features} component identifies predictors with
#' estimable coefficients in the fitted model. It does not represent
#' automatic feature selection.
#'
#' Logistic regression may encounter convergence problems, particularly
#' under complete or quasi-complete separation and strong multicollinearity.
#' Original model warnings are retained, and convergence status is
#' reported in the output.
#'
#' Classification performance metrics are accompanied by confidence intervals
#' estimated using non-parametric bootstrap resampling of the held-out test
#' samples. For binary classification, the AUC confidence interval is estimated
#' using the DeLong method.
#'
#' This function currently supports binary classification only.
#' Multinomial logistic regression is not supported.
#'
#' @examples
#' # Example: Train a Logistic Regression algorithm using the iris dataset
#' # to discriminate between versicolor and virginica species.
#'
#' # Select species
#' iris_data <- iris[
#'   iris$Species %in% c("versicolor", "virginica"),
#' ]
#'
#' # Create columns "Sample" and "Species"
#' iris_data$Sample <- rownames(iris_data)
#' iris_data$Species <- droplevels(iris_data$Species)
#'
#' # Train and test model
#' logistic_model <- learn_logistic(
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
#' @references
#' Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013).
#' \emph{Applied Logistic Regression} (3rd ed.). Wiley.
#'
#' @seealso
#' \code{\link[stats]{glm}},
#' \code{\link{learn_elastic_net}},
#' \code{\link{learn_tree}}
#'
#' @export

learn_logistic <- function(
    data,
    target_var,
    feature_subset = NULL,
    sample_col = "Sample",
    train_fraction = 0.7,
    positive_class = NULL,
    threshold = 0.5,
    seed = 123,
    plot = TRUE
) {

# =============================================================================
# 0. Basic checks
# =============================================================================

.check_dependencies(c(
  "ggplot2",
  "pROC",
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

supported <- vapply(
  X,
  function(x) {
    is.numeric(x) ||
      is.integer(x) ||
      is.factor(x) ||
      is.logical(x) ||
      is.character(x)
  },
  logical(1)
)

if (any(!supported)) {

  stop(
    "All predictors must be numeric or categorical. ",
    "Unsupported variables detected: ",
    paste(
      colnames(X)[!supported],
      collapse = ", "
    )
  )

}

# Convert categorical variables to factors
categorical <- vapply(
  X,
  function(x) {
    is.factor(x) ||
      is.logical(x) ||
      is.character(x)
  },
  logical(1)
)

X[categorical] <- lapply(
  X[categorical],
  factor
)

# Identify predictor types
predictor_types <- .get_predictor_types(X)

# =============================================================================
# 3. Parameter validation
# =============================================================================

# Train fraction
if (
  length(train_fraction) != 1 ||
  !is.numeric(train_fraction) ||
  !is.finite(train_fraction) ||
  train_fraction <= 0 ||
  train_fraction >= 1
) {
  stop("'train_fraction' must be a numeric value between 0 and 1.")
}

# Threshold
if (
  length(threshold) != 1 ||
  !is.numeric(threshold) ||
  !is.finite(threshold) ||
  threshold <= 0 ||
  threshold >= 1
) {
  stop("'threshold' must be a single numeric value between 0 and 1.")
}

# Seed
if (
  length(seed) != 1 ||
  !is.numeric(seed) ||
  !is.finite(seed) ||
  seed != as.integer(seed)
) {
  stop("'seed' must be a single integer.")
}

# Plot
if (
  !is.logical(plot) ||
  length(plot) != 1 ||
  is.na(plot)
) {
  stop("'plot' must be TRUE or FALSE.")
}

# Positive class
if (!is.null(positive_class)) {

  if (
    !is.character(positive_class) ||
    length(positive_class) != 1 ||
    is.na(positive_class)
  ) {
    stop("'positive_class' must be NULL or a single character value.")
  }

  if (!positive_class %in% levels(y)) {
    stop("'positive_class' must match one of the target classes.")
  }

}

# Logistic regression: binary classification only
if (nlevels(y) != 2) {

  stop(
    "'learn_logistic()' currently supports binary classification only. ",
    "Multinomial logistic regression will be implemented separately."
  )

}

# =============================================================================
# 4. Train / Test split
# =============================================================================

set.seed(seed)

split_idx <- .make_stratified_split(
  y = y,
  p = train_fraction
)

train_idx <- split_idx$train
test_idx <- split_idx$test

X_train <- X[train_idx, , drop = FALSE]
X_test  <- X[test_idx, , drop = FALSE]

y_train <- y[train_idx]
y_test  <- y[test_idx]

# =============================================================================
# 5. Prepare response variable
# =============================================================================

if (is.null(positive_class)) {
  positive_class <- levels(y_train)[2]
}

negative_class <- setdiff(
  levels(y_train),
  positive_class
)

y_train_model <- factor(
  y_train,
  levels = c(negative_class, positive_class)
)

y_test_model <- factor(
  y_test,
  levels = c(negative_class, positive_class)
)

# =============================================================================
# 6. Encode predictors
# =============================================================================

encoding_formula <- stats::reformulate(
  termlabels = colnames(X_train)
)

encoding_terms <- stats::terms(
  encoding_formula,
  data = X_train
)

# Store categorical levels
factor_levels <- lapply(
  X_train[
    vapply(X_train, is.factor, logical(1))
  ],
  levels
)

# Generate training matrix
X_train_matrix <- stats::model.matrix(
  encoding_terms,
  data = X_train
)

X_train_matrix <- X_train_matrix[
  ,
  colnames(X_train_matrix) != "(Intercept)",
  drop = FALSE
]

# Generate test matrix using training structure
X_test_matrix <- stats::model.matrix(
  encoding_terms,
  data = X_test,
  xlev = factor_levels
)

X_test_matrix <- X_test_matrix[
  ,
  colnames(X_test_matrix) != "(Intercept)",
  drop = FALSE
]

if (!identical(
  colnames(X_train_matrix),
  colnames(X_test_matrix)
)) {

  stop(
    "Training and test matrices have inconsistent predictor columns."
  )

}

# Number of encoded predictors before filtering
n_features_encoded <- ncol(X_train_matrix)

# =============================================================================
# 7. Remove zero-variance predictors
# =============================================================================

predictor_sd <- apply(
  X_train_matrix,
  2,
  stats::sd
)

valid_features <- is.finite(predictor_sd) &
  predictor_sd > 0

if (!any(valid_features)) {
  stop("No predictors with non-zero variance were found.")
}

removed_features <- colnames(X_train_matrix)[
  !valid_features
]

X_train_matrix <- X_train_matrix[
  ,
  valid_features,
  drop = FALSE
]

X_test_matrix <- X_test_matrix[
  ,
  valid_features,
  drop = FALSE
]

# =============================================================================
# 8. Prepare model data
# =============================================================================

train_model_data <- data.frame(
  .response = y_train_model,
  X_train_matrix,
  check.names = FALSE
)

test_model_data <- as.data.frame(
  X_test_matrix,
  check.names = FALSE
)

# =============================================================================
# 9. Train logistic regression
# =============================================================================

model <- stats::glm(
  .response ~ .,
  data = train_model_data,
  family = stats::binomial(link = "logit"),
  na.action = stats::na.fail
)

# =============================================================================
# 10. Model diagnostics
# =============================================================================

model_summary <- summary(model)

converged <- isTRUE(model$converged)

if (!converged) {
  warning(
    "Logistic regression did not converge. ",
    "Model estimates may be unreliable."
  )
}

# Check aliased coefficients
aliased_coefficients <- names(
  stats::coef(model)
)[
  is.na(stats::coef(model))
]

# =============================================================================
# 11. Generate predictions
# =============================================================================

pred_prob_positive <- stats::predict(
  model,
  newdata = test_model_data,
  type = "response"
)

pred_prob <- cbind(
  1 - pred_prob_positive,
  pred_prob_positive
)

colnames(pred_prob) <- c(
  negative_class,
  positive_class
)

rownames(pred_prob) <- rownames(X_test_matrix)

# =============================================================================
# 12. Predicted classes
# =============================================================================

pred_class <- ifelse(
  pred_prob[, positive_class] >= threshold,
  positive_class,
  negative_class
)

pred_class <- factor(
  pred_class,
  levels = levels(y)
)

# =============================================================================
# 13. Model performance
# =============================================================================

performance <- .classification_metrics(
  predicted = pred_class,
  observed = y_test,
  n_boot = 2000,
  conf_level = 0.95,
  seed = seed + 1
)

cm <- performance$confusion_matrix
cm_tab <- performance$confusion_table
metrics <- performance$metrics
accuracy <- metrics$accuracy
confidence_intervals <- performance$confidence_intervals

# =============================================================================
# 14. ROC and AUC
# =============================================================================

auc_ci <- NULL

roc_results <- .classification_roc(
  observed = y_test,
  probabilities = pred_prob,
  positive_class = positive_class
)

roc_obj <- roc_results$roc
auc_val <- roc_results$auc

auc_ci_raw <- roc_results$auc_ci

auc_ci <- list(
  lower = unname(auc_ci_raw[1]),
  estimate = unname(auc_ci_raw[2]),
  upper = unname(auc_ci_raw[3]),
  method = "delong"
)

metrics$auc <- auc_val
metrics$auc_ci <- auc_ci

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

# =============================================================================
# 15. Extract coefficients and odds ratios
# =============================================================================

# Extract all model coefficients
coef_estimates <- stats::coef(model)

# Extract coefficient statistics
coef_summary <- summary(model)$coefficients

# Build coefficient table
coef_df <- data.frame(
  Feature = names(coef_estimates),
  Coefficient = as.numeric(coef_estimates),
  Std_Error = NA_real_,
  Z_value = NA_real_,
  P_value = NA_real_,
  stringsAsFactors = FALSE
)

# Match estimable coefficients to summary statistics
coef_idx <- match(
  rownames(coef_summary),
  coef_df$Feature
)

coef_df$Std_Error[coef_idx] <- coef_summary[, "Std. Error"]
coef_df$Z_value[coef_idx] <- coef_summary[, "z value"]
coef_df$P_value[coef_idx] <- coef_summary[, "Pr(>|z|)"]

# Calculate odds ratios
coef_df$Odds_Ratio <- exp(
  coef_df$Coefficient
)

# Wald 95% confidence intervals
z_critical <- stats::qnorm(0.975)

coef_df$OR_CI_Lower <- exp(
  coef_df$Coefficient -
    z_critical * coef_df$Std_Error
)

coef_df$OR_CI_Upper <- exp(
  coef_df$Coefficient +
    z_critical * coef_df$Std_Error
)

# Identify estimable coefficients
coef_df$Estimable <- !is.na(
  coef_df$Coefficient
) & is.finite(coef_df$Coefficient)

# Remove intercept for feature-level results
feature_coef <- coef_df[
  coef_df$Feature != "(Intercept)",
  ,
  drop = FALSE
]

# Absolute coefficient magnitude
feature_coef$Abs_Coefficient <- abs(
  feature_coef$Coefficient
)

# Sort by absolute coefficient magnitude
feature_coef <- feature_coef[
  order(
    feature_coef$Abs_Coefficient,
    decreasing = TRUE,
    na.last = TRUE
  ),
  ,
  drop = FALSE
]

rownames(feature_coef) <- NULL

# Identify estimable features
selected_features <- unique(
  feature_coef$Feature[
    feature_coef$Estimable
  ]
)

selected_features <- data.frame(
  Feature = selected_features,
  stringsAsFactors = FALSE
)

# Extract odds ratios separately
odds_ratios <- feature_coef[
  ,
  c(
    "Feature",
    "Odds_Ratio",
    "OR_CI_Lower",
    "OR_CI_Upper",
    "P_value"
  ),
  drop = FALSE
]

rownames(odds_ratios) <- NULL

# =============================================================================
# 16. Feature diagnostics
# =============================================================================

n_features_original <- n_features_encoded

n_features_after_filter <- ncol(X_train_matrix)

n_features_estimable <- sum(
  feature_coef$Estimable
)

feature_reduction <- if (n_features_original > 0) {

  (
    n_features_original -
      n_features_estimable
  ) / n_features_original * 100

} else {

  NA_real_

}

feature_diagnostics <- list(
  n_original = n_features_original,
  n_zero_variance_removed = length(removed_features),
  n_after_filter = n_features_after_filter,
  n_estimable = n_features_estimable,
  n_aliased = length(aliased_coefficients),
  n_removed = n_features_original - n_features_estimable,
  reduction_percent = feature_reduction
)


# =============================================================================
# 17. Generate plots
# =============================================================================

plots <- list()
combined_plot <- NULL

if (plot) {

  # Confusion matrix
  plots$confusion_matrix <- .plot_confusion_matrix(
    confusion_table = cm_tab
  )

  # ROC curve
  plots$roc_curve <- .plot_roc_curve(
    roc_object = roc_obj,
    auc_value = auc_val
  )

  # Combine plots
  combined_plot <- .combine_classification_plots(
    plots = plots,
    ncol = 2
  )

  print(combined_plot)

}

# =============================================================================
# 18. Build frozen model
# =============================================================================

frozen_model <- list(

  # Schema information
  schema_version = 1L,

  # Identification
  algorithm = "logistic",
  model = model,
  model_family = "binomial",

  # Predictor information
  predictor_names = colnames(X),
  predictor_types = predictor_types,

  # Training data
  training_data = list(
    predictors = X_train,
    response = y_train,
    model_matrix = X_train_matrix
  ),

  preprocessing = list(
    method = "model_matrix",
    encoding_terms = encoding_terms,
    factor_levels = factor_levels,
    retained_columns = colnames(X_train_matrix),
    removed_features = removed_features
  ),

  # Response information
  response = list(
    target_var = target_var,
    target_levels = levels(y),
    positive_class = positive_class,
    negative_class = negative_class
  ),

  # Prediction settings
  prediction = list(
    threshold = threshold
  ),

  # Algorithm-specific parameters
  algorithm_parameters = list(
    link = "logit"
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
  "autolearn_logistic",
  "list"
)

# =============================================================================
# 19. Output
# =============================================================================

# Build prediction data with levels
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

obj <- list(

  # Identification
  call = match.call(),
  algorithm = "logistic_regression",
  model_family = "binomial",

  # Fitted models
  model = model,
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
    coefficients = feature_coef,
    all_coefficients = coef_df,
    odds_ratios = odds_ratios,
    selected_features = selected_features,
    feature_diagnostics = feature_diagnostics,
    removed_features = removed_features,
    aliased_coefficients = aliased_coefficients,
    converged = converged
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
    positive_class = positive_class,
    threshold = threshold,
    seed = seed,
    plot = plot
  )
)

class(obj) <- "learn_logistic"

print(obj)

return(invisible(obj))

}

# =============================================================================
# Print method for learn_logistic objects
# =============================================================================
#' Print method for learn_logistic objects
#'
#' Displays a formatted summary of a \code{learn_logistic} object.
#'
#' @param x An object of class \code{learn_logistic}.
#' @param ... Further arguments (ignored).
#' @return Invisibly returns \code{x}.
#' @method print learn_logistic
#' @export

print.learn_logistic <- function(x, ...) {

  # Get confidence intervals
  get_ci <- function(metric) {
    .get_ci(
      confidence_intervals = x$confidence_intervals,
      metric = metric
    )
  }

  .print_header("Logistic Regression Results")

  # General information
  n_train <- length(x$train_samples)
  n_test <- length(x$test_samples)
  n_feat <- length(x$features)

  classes <- paste(
    unique(x$prediction_data$Observed),
    collapse = " vs "
  )

  n_estimable <- x$details$feature_diagnostics$n_estimable

  # ===============================================
  # Data and setup
  # ===============================================

  .print_block("Data & Setup", function() {
    cat(
      "Target Variable:           ",
      x$parameters$target_var,
      " (", classes, ")\n",
      sep = ""
    )

    cat("Model Family:              Binomial\n")
    cat("Link Function:             Logit\n")

    cat(
      "Samples (Train/Test):     ",
      n_train,
      "/",
      n_test,
      "\n"
    )

    cat(
      "Features Available:       ",
      n_feat,
      "\n"
    )

    cat(
      "Positive Class:           ",
      x$parameters$positive_class,
      "\n"
    )

    cat(
      "Classification Threshold: ",
      x$parameters$threshold,
      "\n"
    )

  })

  # ===============================================
  # Model diagnostics
  # ===============================================

  .print_block("Model Diagnostics", function() {

    cat(
      "Converged:                ",
      x$details$converged,
      "\n"
    )

    cat(
      "Encoded Predictors:       ",
      x$details$feature_diagnostics$n_original,
      "\n"
    )

    cat(
      "Zero-Variance Removed:    ",
      x$details$feature_diagnostics$n_zero_variance_removed,
      "\n"
    )

    cat(
      "Estimable Coefficients:   ",
      n_estimable,
      "\n"
    )

    cat(
      "Aliased Coefficients:     ",
      x$details$feature_diagnostics$n_aliased,
      "\n"
    )

    cat(
      "Features Removed:         ",
      x$details$feature_diagnostics$n_removed,
      "\n"
    )

  })

  # ===============================================
  # Performance
  # ===============================================

  .print_block("Performance (Test Set; estimates with 95% CI)", function() {

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
  })

  invisible(x)

}
