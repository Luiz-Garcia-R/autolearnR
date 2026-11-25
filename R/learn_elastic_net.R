# =============================================================================
# LEARN ELASTIC NET
# =============================================================================

#' Elastic Net Classification
#'
#' Trains an Elastic Net classification model using a single data frame
#' containing a categorical target variable and one or more predictor
#' variables.
#'
#' The function provides a general framework for classification tasks using
#' clinical, phenotypic, molecular, or other quantitative features. Predictors
#' can be selected explicitly using \code{feature_subset}, allowing the same
#' function to be used regardless of the biological or clinical origin of
#' the data.
#'
#' The function performs stratified train/test splitting, predictor encoding,
#' zero-variance filtering, Elastic Net regularization, cross-validation for
#' lambda selection, test-set evaluation, and coefficient-based feature
#' selection. Diagnostic plots can also be generated automatically.
#'
#' Both binary and multiclass classification are supported through binomial
#' and multinomial models, respectively.
#'
#' @param data Data frame containing the target variable and predictor
#'   variables. Each row represents a sample and each column represents a
#'   variable. The data frame may contain clinical, phenotypic, molecular, or
#'   other quantitative features.
#'
#' @param target_var Character. Name of the categorical outcome variable to be
#'   predicted. The variable is converted to a factor if necessary.
#'
#' @param feature_subset Optional character vector specifying the predictor
#'   variables to include in the model. If \code{NULL}, all available
#'   predictor columns are used, excluding \code{target_var} and
#'   \code{sample_col}. This allows the same interface to be used for
#'   metadata-only, molecular-only, or combined datasets.
#'
#' @param sample_col Character. Optional column containing sample identifiers
#'   (default: \code{"Sample"}). This column is excluded from the predictors.
#'   If the column is not present, a warning is issued and sample identifiers
#'   are not stored separately.
#'
#' @param train_fraction Numeric. Proportion of samples assigned to the
#'   training set using stratified sampling (default: \code{0.7}).
#'
#' @param alpha Numeric. Elastic Net mixing parameter between Ridge and LASSO
#'   regularization (default: \code{0.5}).
#'   Values closer to \code{0} increase the Ridge component, whereas values
#'   closer to \code{1} increase the LASSO component.
#'   A value of \code{1} corresponds to LASSO, while \code{0} corresponds
#'   to Ridge regression.
#'
#' @param nfolds Integer. Number of folds used for stratified
#'   cross-validation during lambda selection (default: \code{5}).
#'   Each class in the training set must contain at least \code{nfolds}
#'   samples.
#'
#' @param type_measure Character. Performance measure used to select the
#'   optimal regularization parameter during cross-validation
#'   (default: \code{"deviance"}).
#'   For binary classification, supported measures are
#'   \code{"deviance"}, \code{"class"}, \code{"auc"}, \code{"mse"},
#'   and \code{"mae"}.
#'   For multinomial classification, supported measures are
#'   \code{"deviance"}, \code{"class"}, \code{"mse"}, and \code{"mae"}.
#'
#' @param lambda_choice Character. Criterion used to select the final
#'   regularization parameter (default: \code{"1se"}).
#'   Available options are:
#'   \describe{
#'     \item{\code{"1se"}}{Selects the largest lambda within one standard
#'       error of the minimum cross-validation error, generally producing
#'       a more regularized and parsimonious model.}
#'     \item{\code{"min"}}{Selects the lambda associated with the minimum
#'       cross-validation error.}
#'   }
#'
#' @param seed Integer. Random seed used for reproducible data partitioning,
#'   cross-validation fold assignment, and model fitting
#'   (default: \code{123}).
#'
#' @param positive_class Optional character. Reference class used for binary
#'   classification, probability interpretation, and ROC/AUC calculations.
#'   If \code{NULL}, the second level of the target factor is used.
#'   This parameter is only applicable to binary classification.
#'
#' @param plot Logical. Whether to generate and print the combined diagnostic
#'   plots to the graphics device (default: \code{TRUE}).
#'
#' @return
#' Invisibly returns an object of class \code{learn_elastic_net}.
#'
#' The returned object contains:
#' \describe{
#'   \item{call}{Original function call.}
#'   \item{algorithm}{Algorithm identifier (\code{"elastic_net"}).}
#'   \item{model_family}{Classification family (\code{"binomial"} or
#'     \code{"multinomial"}).}
#'   \item{model}{Fitted cross-validation object returned by
#'     \code{glmnet::cv.glmnet}.}
#'   \item{frozen_model}{Self-contained model object for subsequent
#'     predictions on new data.}
#'   \item{features}{Names of the original predictors used in the model.}
#'   \item{train_samples, test_samples}{Sample identifiers assigned
#'     to the training and test sets.}
#'   \item{prediction_data}{Data frame containing sample identifiers,
#'     observed classes, and predicted classes.}
#'   \item{predictions}{Predicted class labels for the test set.}
#'   \item{probabilities}{Predicted class probabilities for each
#'     target class.}
#'   \item{confusion_matrix}{Detailed confusion matrix.}
#'   \item{confusion_table}{Confusion matrix in tabular format.}
#'   \item{metrics}{Classification performance metrics.}
#'   \item{roc}{ROC object for binary classification, otherwise
#'     \code{NULL}.}
#'   \item{confidence_intervals}{A list containing confidence intervals
#'   for the classification performance metrics, estimated using
#'   non-parametric bootstrap resampling.}
#'   \item{details}{List containing algorithm-specific results,
#'     including coefficients, selected features, feature diagnostics,
#'     removed features, cross-validation results, and fold distribution.}
#'   \item{plots}{List containing individual diagnostic plots.}
#'   \item{combined_plot}{Combined diagnostic visualization, or
#'     \code{NULL} when plotting is disabled.}
#'   \item{parameters}{List containing the main analysis parameters.}
#' }
#'
#' @details
#' \strong{Data structure:}
#'   The function expects a sample-by-variable data frame. The target
#'   variable and all predictors must be contained in the same object.
#'   This design keeps the Elastic Net implementation independent of the
#'   biological origin of the predictors. For example, age, sex, gene
#'   expression, protein abundance, or metabolite measurements can all
#'   be supplied through the same interface.
#'
#' \strong{Predictor selection:}
#'   If \code{feature_subset} is supplied, only the specified columns are
#'   used as predictors. If it is \code{NULL}, all columns other than
#'   \code{target_var} and \code{sample_col} are used.
#'   Predictors may be numeric or categorical. Categorical variables,
#'   including factors, logical variables, and character variables, are
#'   converted to factors and encoded into a numeric design matrix using
#'   \code{model.matrix}.
#'
#' \strong{Sample identifiers:}
#'   If \code{sample_col} is present, its values are used to track samples
#'   through the analysis and are excluded from the predictor matrix.
#'   Duplicate sample identifiers are not allowed.
#'
#' \strong{Stratification:}
#'   The train/test partition is performed using stratified sampling,
#'   preserving the class structure of the target variable across the
#'   two sets. Cross-validation folds are also assigned using stratified
#'   sampling, ensuring that each fold contains observations from every
#'   target class.
#'
#' \strong{Predictor encoding and filtering:}
#'   Categorical predictors are converted into dummy variables using
#'   the training data structure. The same encoding is applied to the
#'   test set to ensure consistent predictor columns.
#'   Predictors with zero variance in the training set are removed before
#'   model fitting. The same filtering is applied to the test set.
#'
#' \strong{Elastic Net regularization:}
#'   Model fitting is performed using \code{glmnet::cv.glmnet}.
#'   The \code{alpha} parameter controls the balance between Ridge and
#'   LASSO regularization, while lambda controls the overall strength
#'   of the penalty. Predictors are standardized internally by
#'   \code{glmnet} during model fitting.
#'
#' \strong{Lambda selection:}
#'   The regularization parameter is selected through cross-validation.
#'   With \code{lambda_choice = "min"}, the lambda associated with the
#'   minimum cross-validation error is selected. With
#'   \code{lambda_choice = "1se"}, the largest lambda within one standard
#'   error of the minimum is selected, favoring a more regularized model.
#'
#' \strong{Classification:}
#'   Binary classification is performed using a binomial model, whereas
#'   problems involving more than two classes use a multinomial model.
#'   For binary classification, the positive class is used to define
#'   predicted probabilities and class assignments, with a default
#'   classification threshold of \code{0.5}.
#'   For multiclass classification, the predicted class corresponds
#'   to the class with the highest estimated probability.
#'
#' \strong{Model evaluation:}
#'   Model performance is evaluated on the held-out test set using a
#'   confusion matrix, Accuracy, Kappa, and class-wise Precision, Recall,
#'   Specificity, and F1 score. For binary classification, an ROC curve
#'   and AUC are also calculated.
#'
#' \strong{Feature selection:}
#'   Features are selected based on non-zero coefficients in the final
#'   model at the selected lambda. Coefficient magnitude can be used to
#'   rank predictors according to their contribution to the fitted linear
#'   predictor. In multinomial models, coefficients are estimated
#'   separately for each class.
#'
#'   The reported feature reduction percentage is calculated from the
#'   number of encoded predictors before filtering and the number of
#'   unique predictors retained with non-zero coefficients.
#'
#' \strong{Visualization:}
#'   When \code{plot = TRUE}, the function generates a composite diagnostic
#'   visualization using \pkg{ggplot2} and \pkg{patchwork}. The combined
#'   visualization includes the confusion matrix, cross-validation curve,
#'   and coefficient paths. For binary classification, an ROC curve is
#'   also included. All generated plots are stored in the returned object's
#'   \code{plots} component. When \code{plot = FALSE}, no plots are generated.
#'
#' Classification performance metrics are accompanied by confidence intervals
#' estimated using non-parametric bootstrap resampling of the held-out test
#' samples. For binary classification, the AUC confidence interval is estimated
#' using the DeLong method.
#'
#' @examples
#' # Example: Train a Elastic Net algorithm using the iris dataset
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
#' elastic_model <- learn_elastic_net(
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
#' @importFrom stats predict model.matrix reformulate terms sd coef
#' @importFrom stats setNames
#'
#' @export

learn_elastic_net <- function(
    data,
    target_var,
    feature_subset = NULL,
    sample_col = "Sample",
    train_fraction = 0.7,
    alpha = 0.5,
    nfolds = 5,
    type_measure = c("deviance",  "class", "auc", "mse"),
    lambda_choice = c("1se", "min"),
    seed = 123,
    positive_class = NULL,
    plot = TRUE
) {

# =============================================================================
# 0. Basic checks
# =============================================================================

.check_dependencies(c(
  "glmnet",
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

# Alpha
if (
  length(alpha) != 1 ||
  !is.numeric(alpha) ||
  !is.finite(alpha) ||
  alpha < 0 ||
  alpha > 1
) {
  stop("'alpha' must be a single numeric value between 0 and 1.")
}

# Number of folds
if (
  length(nfolds) != 1 ||
  !is.numeric(nfolds) ||
  !is.finite(nfolds) ||
  nfolds < 3 ||
  nfolds != as.integer(nfolds)
) {
  stop("'nfolds' must be an integer greater than or equal to 3.")
}

# Type of cross-validation measure
type_measure <- match.arg(type_measure)

# Lambda selection
lambda_choice <- match.arg(lambda_choice)

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


# Check minimum class representation for CV
train_class_counts <- table(y_train)

if (any(train_class_counts < nfolds)) {

  stop(
    "Each class in the training set must contain at least ",
    nfolds,
    " samples for the selected cross-validation strategy. ",
    "Current training class counts: ",
    paste(
      names(train_class_counts),
      train_class_counts,
      sep = " = ",
      collapse = ", "
    ),
    ". Consider reducing 'nfolds' or increasing 'train_fraction'."
  )

}

# =============================================================================
# 5. Identify model family
# =============================================================================

n_classes <- nlevels(y_train)

if (n_classes < 2) {
  stop("The training set must contain at least two classes.")
}

if (n_classes == 2) {
  model_family <- "binomial"

} else {
  model_family <- "multinomial"

}

if (model_family == "multinomial" && !is.null(positive_class)) {
  stop("'positive_class' is only applicable to binary classification.")
}

# Supported CV measures
if (model_family == "binomial") {
  supported_measures <- c(
    "deviance",
    "class",
    "auc",
    "mse",
    "mae"
  )

} else {
  supported_measures <- c(
    "deviance",
    "class",
    "mse",
    "mae"
  )
}

if (!type_measure %in% supported_measures) {

  stop(
    "Unsupported 'type_measure' for ",
    model_family,
    " models. Available options: ",
    paste(supported_measures, collapse = ", "),
    "."
  )
}

# =============================================================================
# 6. Encode predictors
# =============================================================================

# Build formula
encoding_formula <- stats::reformulate(
  termlabels = colnames(X_train)
)

# Create model terms
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

# Remove intercept
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

# Remove intercept
X_test_matrix <- X_test_matrix[
  ,
  colnames(X_test_matrix) != "(Intercept)",
  drop = FALSE
]

# Ensure identical predictor structure
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
# 8. Prepare response variable
# =============================================================================

if (model_family == "binomial") {

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

} else {

  y_train_model <- y_train
  y_test_model <- y_test

}

# =============================================================================
# 9. Stratified cross-validation folds
# =============================================================================

set.seed(seed + 1L)

foldid <- integer(length(y_train_model))

for (class_name in levels(y_train_model)) {

  class_idx <- which(
    y_train_model == class_name
  )

  class_folds <- sample(
    rep(
      seq_len(nfolds),
      length.out = length(class_idx)
    )
  )

  foldid[class_idx] <- class_folds
}

# Validate fold distribution
fold_distribution <- table(
  Fold = foldid,
  Class = y_train_model
)

if (any(fold_distribution == 0)) {

  stop(
    "Stratified cross-validation failed: ",
    "at least one fold does not contain all classes."
  )
}

# =============================================================================
# 10. Train Elastic Net with cross-validation
# =============================================================================

set.seed(seed + 2L)

cv_fit <- glmnet::cv.glmnet(
  x = X_train_matrix,
  y = y_train_model,
  family = model_family,
  alpha = alpha,
  type.measure = type_measure,
  nfolds = nfolds,
  foldid = foldid,
  standardize = TRUE,
  intercept = TRUE
)

# =============================================================================
# 11. Select optimal lambda
# =============================================================================

lambda_min <- cv_fit$lambda.min
lambda_1se <- cv_fit$lambda.1se

selected_lambda <- if (lambda_choice == "1se") {
  lambda_1se
} else {
  lambda_min
}

# Extract selected lambda index
lambda_index <- which.min(
  abs(cv_fit$lambda - selected_lambda)
)

# =============================================================================
# 12. Final model
# =============================================================================

model_used <- cv_fit$glmnet.fit

# =============================================================================
# 13. Generate predictions
# =============================================================================

pred_raw <- predict(
  cv_fit,
  newx = X_test_matrix,
  s = selected_lambda,
  type = "response"
)

# Binary classification
if (model_family == "binomial") {

  positive_prob <- as.numeric(pred_raw)

  pred_prob <- cbind(
    1 - positive_prob,
    positive_prob
  )

  colnames(pred_prob) <- c(
    negative_class,
    positive_class
  )

  rownames(pred_prob) <- rownames(X_test_matrix)

} else {

  # Multiclass classification
  pred_prob <- matrix(
    as.numeric(pred_raw),
    nrow = nrow(X_test_matrix),
    ncol = length(levels(y_train_model)),
    dimnames = list(
      rownames(X_test_matrix),
      levels(y_train_model)
    )
  )
}

# =============================================================================
# 14. Predicted classes
# =============================================================================

if (model_family == "binomial") {

  pred_class <- ifelse(
    pred_prob[, positive_class] >= 0.5,
    positive_class,
    negative_class
  )

} else {

  pred_class <- colnames(pred_prob)[
    max.col(
      pred_prob,
      ties.method = "first"
    )
  ]

}

pred_class <- factor(
  pred_class,
  levels = levels(y)
)

# =============================================================================
# 15. Performance evaluation
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
# 16. ROC and AUC
# =============================================================================

roc_results <- NULL
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

}

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
# 17. Extract coefficients
# =============================================================================

coef_model <- stats::coef(
  cv_fit,
  s = selected_lambda
)

# Binary classification
if (model_family == "binomial") {

  coef_matrix <- as.matrix(coef_model)

  coef_df <- data.frame(
    Feature = rownames(coef_matrix),
    Coefficient = as.numeric(coef_matrix[, 1]),
    stringsAsFactors = FALSE
  )

  coef_df$Class <- positive_class

} else {

  # Multiclass classification
  coef_list <- lapply(
    names(coef_model),
    function(class_name) {

      coef_matrix <- as.matrix(
        coef_model[[class_name]]
      )

      data.frame(
        Feature = rownames(coef_matrix),
        Coefficient = as.numeric(coef_matrix[, 1]),
        Class = class_name,
        stringsAsFactors = FALSE
      )

    }
  )

  coef_df <- do.call(
    rbind,
    coef_list
  )
}

# =============================================================================
# 18. Identify selected features
# =============================================================================

coef_df <- coef_df[
  coef_df$Feature != "(Intercept)",
  ,
  drop = FALSE
]

# Absolute coefficient magnitude
coef_df$Abs_Coefficient <- abs(
  coef_df$Coefficient
)

# Identify non-zero coefficients
selected_coef <- coef_df[
  coef_df$Coefficient != 0,
  ,
  drop = FALSE
]

# Sort by absolute coefficient
selected_coef <- selected_coef[
  order(
    selected_coef$Abs_Coefficient,
    decreasing = TRUE
  ),
  ,
  drop = FALSE
]

rownames(selected_coef) <- NULL

# =============================================================================
# 19. Selected features
# =============================================================================

selected_features <- unique(
  selected_coef$Feature
)

selected_features <- data.frame(
  Feature = selected_features,
  stringsAsFactors = FALSE
)

# =============================================================================
# 20. Feature reduction diagnostics
# =============================================================================

n_features_original <- n_features_encoded
n_features_after_filter <- ncol(X_train_matrix)
n_features_selected <- nrow(selected_features)

feature_reduction <- if (n_features_original > 0) {

  (
    n_features_original -
      n_features_selected
  ) / n_features_original * 100

} else {

  NA_real_

}

feature_diagnostics <- list(
  n_original = n_features_original,
  n_zero_variance_removed = length(removed_features),
  n_after_filter = n_features_after_filter,
  n_selected = n_features_selected,
  n_removed = n_features_original - n_features_selected,
  reduction_percent = feature_reduction
)

# Cross-validation results
cv_results <- data.frame(
  Lambda = cv_fit$lambda,
  Log_Lambda = log(cv_fit$lambda),
  CV_Error = cv_fit$cvm,
  CV_SD = cv_fit$cvsd,
  Nonzero_Coefficients = cv_fit$glmnet.fit$df
)

# =============================================================================
# 21. Generate plots
# =============================================================================

plots <- list()
combined_plot <- NULL

if (plot) {

  # Confusion matrix
  plots$confusion_matrix <- .plot_confusion_matrix(
    confusion_table = cm_tab
  )

  # Cross-validation curve
  plots$cv_curve <- .plot_cv_curve(
    cv_results = cv_results,
    lambda_min = lambda_min,
    lambda_1se = lambda_1se,
    lambda_choice = lambda_choice
  )

  # Coefficient paths
  plots$coefficient_paths <- .plot_coefficient_paths(
    model = cv_fit$glmnet.fit,
    model_family = model_family,
    selected_lambda = selected_lambda,
    positive_class = positive_class
  )

  # ROC curve (binary classification only)
  if (model_family == "binomial") {

    plots$roc_curve <- .plot_roc_curve(
      roc_object = roc_obj,
      auc_value = auc_val
    )

  }

  # Combine plots
  combined_plot <- .combine_classification_plots(
    plots = plots,
    ncol = 2
  )

  print(combined_plot)

}

# =============================================================================
# 22. Build frozen model
# =============================================================================

frozen_model <- list(

  # Schema information
  schema_version = 1L,

  # Identification
  algorithm = "elastic_net",
  model = cv_fit,
  model_family = model_family,

  # Training data
  training_data = list(
    predictors = X_train,
    response = y_train,
    model_matrix = X_train_matrix
  ),

  # Predictor information
  predictor_names = colnames(X),
  predictor_types = predictor_types,
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

    positive_class = if (model_family == "binomial") {
      positive_class
    } else {
      NULL
    },

    negative_class = if (model_family == "binomial") {
      negative_class
    } else {
      NULL
    }

  ),

  # Prediction settings
  prediction = list(
    threshold = if (model_family == "binomial") {
      0.5
    } else {
      NULL
    }

  ),

  # Algorithm-specific parameters
  algorithm_parameters = list(
    alpha = alpha,
    type_measure = type_measure,
    lambda_choice = lambda_choice,
    selected_lambda = selected_lambda,
    standardize = TRUE,
    intercept = TRUE,
    nfolds = nfolds
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
  "autolearn_elastic_net",
  "list"
)

# =============================================================================
# 23. Build prediction data with levels
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
# 24. Output
# =============================================================================

obj <- list(

  # Identification
  call = match.call(),
  algorithm = "elastic_net",
  model_family = model_family,

  # Fitted models
  model = cv_fit,
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
    coefficients = selected_coef,
    all_coefficients = coef_df,
    selected_features = selected_features,
    feature_diagnostics = feature_diagnostics,
    removed_features = removed_features,
    cv_results = cv_results,
    fold_distribution = fold_distribution
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
    alpha = alpha,
    nfolds = nfolds,
    type_measure = type_measure,
    lambda_choice = lambda_choice,
    selected_lambda = selected_lambda,
    positive_class = positive_class,
    seed = seed,
    plot = plot
    )
  )

class(obj) <- "learn_elastic_net"

print(obj)

return(invisible(obj))

}

# =============================================================================
# Print S3
# =============================================================================
#' Print method for learn_elastic_net objects
#'
#' Displays a formatted summary of a \code{learn_elastic_net} object.
#'
#' @param x An object of class \code{learn_elastic_net}.
#' @param ... Further arguments (ignored).
#' @return Invisibly returns \code{x}.
#' @method print learn_elastic_net
#' @export

print.learn_elastic_net <- function(x, ...) {

  # Get confidence intervals
  get_ci <- function(metric) {
    .get_ci(
      confidence_intervals = x$confidence_intervals,
      metric = metric
    )
  }

  .print_header("Elastic Net Results")

  # General information
  n_train <- length(x$train_samples)
  n_test  <- length(x$test_samples)
  n_feat  <- length(x$features)

  classes <- paste(
    unique(x$prediction_data$Observed),
    collapse = " vs "
  )

  n_selected <- x$details$feature_diagnostics$n_selected

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
      "Alpha:                 ",
      x$parameters$alpha,
      "\n"
    )

    cat(
      "Lambda Selection:      ",
      x$parameters$lambda_choice,
      "\n"
    )

    cat(
      "Selected Lambda:       ",
      signif(x$parameters$selected_lambda, 5),
      "\n"
    )

    cat(
      "CV Folds:              ",
      x$parameters$nfolds,
      "\n"
    )

  })

  # ===============================================
  # Feature selection and regularization
  # ===============================================

  .print_block("Feature Selection & Regularization", function() {

    cat(
      "Features Before:       ",
      x$details$feature_diagnostics$n_original,
      "\n"
    )

    cat(
      "Zero-Variance Removed: ",
      x$details$feature_diagnostics$n_zero_variance_removed,
      "\n"
    )

    cat(
      "Features Selected:     ",
      n_selected,
      "\n"
    )

    cat(
      "Features Removed:      ",
      x$details$feature_diagnostics$n_removed,
      "\n"
    )

    cat(
      "Feature Reduction:      ",
      round(
        x$details$feature_diagnostics$reduction_percent,
        2
      ),
      "%\n",
      sep = ""
    )

    if (n_selected > 0) {

      cat(
        "Top Feature:           ",
        x$details$coefficients$Feature[1],
        "\n"
      )

    } else {
      cat("Top Feature:           None selected\n")
    }

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

    if (!is.null(x$details$permutation)) {

      cat(
        "Permutation p-value:   ",
        round(x$details$permutation$p_value, 4),
        "\n"
      )

    }
  })

  invisible(x)

}

