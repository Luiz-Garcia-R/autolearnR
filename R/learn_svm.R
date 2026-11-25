# =============================================================================
# LEARN SUPPORT VECTOR MACHINE
# =============================================================================

#' Train and evaluate a Support Vector Machine classification model
#'
#' Trains a Support Vector Machine (SVM) classifier using a stratified
#' training/test split and evaluates its predictive performance on held-out
#' samples.
#'
#' The function supports binary and multiclass classification and provides four
#' kernel functions: \code{"linear"}, \code{"polynomial"}, \code{"radial"}, and
#' \code{"sigmoid"}. Numeric and factor predictors are supported, with
#' categorical predictors automatically encoded into dummy variables during
#' preprocessing.
#'
#' Predictor preprocessing, including dummy encoding and optional
#' standardization, is learned exclusively from the training set and
#' subsequently applied to the test set to avoid data leakage.
#'
#' Classification performance is evaluated using standard metrics and a
#' confusion matrix. For binary classification, ROC analysis and AUC calculation
#' are also performed. An optional permutation test can be used to assess model
#' accuracy under permuted training labels.
#'
#' The function also provides SVM-specific diagnostics, including the number and
#' proportion of support vectors. When exactly two numeric predictors are
#' available, a decision boundary plot can be generated to visualize the
#' classification regions learned by the model.
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
#' @param kernel Character string specifying the kernel function used by the SVM
#'   classifier. Available options are \code{"linear"}, \code{"polynomial"},
#'   \code{"radial"}, and \code{"sigmoid"}. Defaults to \code{"linear"}.
#'
#' @param cost Positive numeric value specifying the cost of constraint
#'   violation (\eqn{C}) in the SVM optimization. Higher values impose
#'   a stronger penalty on classification errors and generally allow
#'   less tolerance for observations violating the margin. The parameter has no
#'   fixed upper bound and should be selected according to the scale and
#'   characteristics of the data. Defaults to \code{1}.
#'
#' @param gamma Positive numeric value controlling the influence of individual
#'   observations in the polynomial, radial, and sigmoid kernels. If
#'   \code{NULL} (default), gamma is automatically calculated as the inverse of
#'   the number of processed predictor variables.
#'
#' @param degree Integer specifying the polynomial degree of the polynomial
#'   kernel. Only relevant when \code{kernel = "polynomial"}.
#'   Defaults to \code{3}.
#'
#' @param coef0 Numeric value specifying the independent term used in the
#'   polynomial and sigmoid kernels. Only relevant for these kernel functions.
#'   Defaults to \code{0}.
#'
#' @param scale Logical value indicating whether numeric predictors should be
#'   standardized during preprocessing. Scaling parameters are estimated from
#'   the training set and applied to the test set. Defaults to \code{TRUE}.
#'
#' @param seed Integer specifying the random seed for reproducibility.
#'   Defaults to \code{123}.
#'
#' @param positive_class Optional character string specifying the positive
#'   class for binary classification. If \code{NULL}, the second target
#'   factor level is used as the positive class. Not applicable to
#'   multiclass classification.
#'
#' @param permutation_test Logical value indicating whether to perform a
#'   permutation test to evaluate model accuracy under permuted training
#'   labels. Defaults to \code{FALSE}.
#'
#' @param nperm Integer specifying the number of permutations used in the
#'   permutation test. Defaults to \code{1000}.
#'
#' @param plot Logical value indicating whether to generate diagnostic
#'   plots. Defaults to \code{TRUE}.
#'
#' @details
#' The function performs the following steps:
#' \enumerate{
#'   \item Prepares the classification dataset and validates the predictors.
#'   \item Splits the data into training and test sets using stratified
#'   sampling.
#'   \item Learns predictor preprocessing parameters from the training set,
#'   including categorical dummy encoding and optional standardization.
#'   \item Applies the learned preprocessing transformations to the test set.
#'   \item Trains a Support Vector Machine classifier using
#'   \code{e1071::svm()} with the selected kernel and hyperparameters.
#'   \item Predicts class labels and class probabilities for the test set.
#'   \item Calculates classification metrics and confusion matrix.
#'   \item Performs ROC analysis and calculates AUC for binary classification.
#'   \item Extracts SVM diagnostics, including the number and proportion
#'   of support vectors.
#'   \item Optionally performs a permutation test.
#'   \item Generates individual diagnostic plots and a combined plot when
#'   \code{plot = TRUE}.
#' }
#'
#' The SVM classifier identifies a decision boundary that separates observations
#' belonging to different classes while maximizing the margin between them.
#' Observations that contribute to defining this boundary are referred to as
#' support vectors.
#'
#' The selected kernel determines how the decision boundary is represented:
#' \itemize{
#'   \item \code{"linear"}: Uses a linear decision boundary in the original
#'   predictor space.
#'   \item \code{"polynomial"}: Allows nonlinear decision boundaries
#'   through polynomial transformations of the predictor space.
#'   \item \code{"radial"}: Uses the radial basis function (RBF) kernel
#'   to capture nonlinear relationships.
#'   \item \code{"sigmoid"}: Uses a sigmoid kernel to represent nonlinear
#'   decision boundaries.
#' }
#'
#' The \code{cost} parameter controls the trade-off between maximizing the
#' classification margin and penalizing classification errors. The \code{gamma}
#' parameter controls the influence of individual observations in nonlinear
#' kernels, while \code{degree} and \code{coef0} provide additional control over
#' polynomial and sigmoid kernels.
#'
#' When \code{gamma = NULL}, its effective value is calculated as
#' \code{1 / ncol(x_train_processed)}, where
#' \code{x_train_processed} represents the processed training matrix. Both the
#' user-supplied and effective gamma values are retained in the returned object.
#'
#' The preprocessing procedure supports numeric and factor predictors. Factor
#' predictors are converted into dummy variables using the training data levels.
#' Numeric predictors with zero variance after preprocessing are excluded. The
#' resulting transformations are stored in the frozen model to support
#' consistent application to new data.
#'
#' The \code{scale} argument controls standardization of numeric predictors
#' during preprocessing. The internal SVM fitting procedure uses the already
#' processed predictor matrix without applying an additional scaling step.
#'
#' The number of support vectors reflects how many training observations
#' contribute to defining the fitted SVM model. Their proportion is calculated
#' relative to the total number of training observations. A larger proportion of
#' support vectors does not necessarily indicate poorer predictive performance.
#'
#' For binary classification, ROC analysis and AUC are calculated using the
#' predicted class probabilities. These probabilities are estimated internally
#' by the SVM implementation and are not necessarily calibrated probabilities.
#'
#' When exactly two numeric predictors are available, a decision boundary plot
#' is generated using the fitted model and training observations. The plot
#' displays predicted classification regions and highlights support vectors. It
#' represents the decision boundary learned from the training set and should not
#' be interpreted as a visualization of test-set performance. The decision
#' boundary plot is not generated for datasets with more than two predictors or
#' categorical predictors.
#'
#' When \code{permutation_test = TRUE}, training labels are repeatedly shuffled
#' while the test labels remain unchanged. The SVM is refitted using each
#' permuted training set, and accuracy is evaluated on the original test set.
#' The empirical p-value is calculated using a correction based on the number of
#' permutations and represents the proportion of permuted accuracies equal to or
#' greater than the observed accuracy, including the correction term.
#'
#' This permutation procedure evaluates predictive performance conditionally on
#' the original training/test partition. Its validity relies on the
#' exchangeability of training labels under the null hypothesis. For datasets
#' with repeated measures, clustered samples, or other dependencies,
#' unrestricted label permutation may not be appropriate.
#'
#' Classification performance metrics are accompanied by confidence intervals
#' estimated using non-parametric bootstrap resampling of the held-out test
#' samples. For binary classification, the AUC confidence interval is estimated
#' using the DeLong method.
#'
#' The returned object follows the common output structure used by
#' \code{autolearnR} classification algorithms. Algorithm-specific
#' results are stored in the \code{details} component, while shared
#' evaluation results are made available through standardized components.
#'
#' @return An object of class \code{"learn_svm"}, containing:
#' \describe{
#'   \item{call}{The original function call.}
#'   \item{algorithm}{Character string identifying the algorithm:
#'   \code{"svm"}.}
#'   \item{model_family}{Classification type, either \code{"binomial"} or
#'   \code{"multinomial"}.}
#'   \item{model}{The fitted SVM model returned by
#'   \code{e1071::svm()}.}
#'   \item{frozen_model}{A model bundle containing the fitted model and
#'   associated metadata, including predictor names, factor levels,
#'   preprocessing parameters, target levels, and SVM hyperparameters.}
#'   \item{features}{Character vector containing the original predictor
#'   variables used for model training.}
#'   \item{train_samples}{Sample identifiers assigned to the training set.}
#'   \item{test_samples}{Sample identifiers assigned to the test set.}
#'   \item{prediction_data}{Data frame containing sample identifiers,
#'   observed classes (\code{Observed}), predicted classes
#'   (\code{Predicted}), and predicted class probabilities when available,
#'   for the test set.}
#'   \item{predictions}{Predicted class labels for the test set.}
#'   \item{probabilities}{Predicted class probabilities for the test set.}
#'   \item{decision_values}{Decision values returned by the SVM prediction
#'   procedure.}
#'   \item{confusion_matrix}{Confusion matrix summarizing observed and
#'   predicted classifications.}
#'   \item{confusion_table}{Tabular representation of the confusion matrix.}
#'   \item{metrics}{Named list of classification performance metrics,
#'   including accuracy, sensitivity, specificity, precision, F1-score,
#'   and AUC when applicable.}
#'   \item{roc}{ROC object for binary classification. Returns
#'   \code{NULL} for multiclass classification.}
#'   \item{confidence_intervals}{A list containing confidence intervals
#'   for the classification performance metrics, estimated using
#'   non-parametric bootstrap resampling.}
#'   \item{details}{A list containing SVM-specific results:
#'   \describe{
#'     \item{support_vectors}{List containing the total number of support
#'     vectors, their proportion relative to the training set, the number
#'     of support vectors per class, and their training indices.}
#'     \item{preprocessing}{Preprocessing information used during model
#'     training, including predictor names, factor levels, dummy-variable
#'     encoding, and scaling parameters.}
#'     \item{permutation}{Results of the optional permutation test,
#'     including permuted accuracies, observed accuracy, empirical
#'     p-value, and number of permutations. Returns \code{NULL} when
#'     the test is not performed.}
#'   }}
#'   \item{plots}{A list containing individual diagnostic plots, including
#'   confusion matrix, ROC curve, decision boundary, and permutation
#'   test plots when applicable. Empty when \code{plot = FALSE}.}
#'   \item{combined_plot}{Combined diagnostic plot. Returns \code{NULL}
#'   when \code{plot = FALSE}.}
#'   \item{parameters}{A list containing the parameters used during model
#'   training and evaluation, including both user-supplied and effective
#'   gamma values.}
#' }
#'
#' @examples
#'
#' # Example: Train a SVM algorithm using the iris dataset
#' # to discriminate between versicolor and virginica species.
#'
#'   # Select species
#' iris_data <- iris[
#'   iris$Species %in% c("versicolor", "virginica", "setosa"),
#' ]
#'
#' # Create columns "Sample" and "Species"
#' iris_data$Sample <- rownames(iris_data)
#' iris_data$Species <- droplevels(iris_data$Species)
#'
#' # Train and test model
#' svm_model <- learn_svm(
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
#'   kernel = "linear",
#'   cost = 1,
#'   seed = 123,
#'   plot = TRUE
#' )
#'
#' @seealso
#' \code{\link[e1071]{svm}},
#' \code{\link[pROC]{roc}},
#' \code{\link{learn_random_forest}},
#' \code{\link{learn_tree}},
#' \code{\link{learn_logistic}},
#' \code{\link{learn_elastic_net}}
#'
#' @export

learn_svm <- function(
    data,
    target_var,
    feature_subset = NULL,
    sample_col = "Sample",
    train_fraction = 0.7,
    kernel = c("linear", "polynomial", "radial", "sigmoid"),
    cost = 1,
    gamma = NULL,
    degree = 3,
    coef0 = 0,
    scale = TRUE,
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
  "e1071",
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
# 2. Parameter and predictor validation
# =============================================================================

# Validate kernel
valid_kernels <- c(
  "linear",
  "polynomial",
  "radial",
  "sigmoid"
)

kernel <- match.arg(kernel, choices = valid_kernels)

# Validate cost
if (
  !is.numeric(cost) ||
  length(cost) != 1 ||
  !is.finite(cost) ||
  cost <= 0
) {

  stop("'cost' must be a positive finite number.")

}

# Validate gamma
if (!is.null(gamma)) {

  if (
    !is.numeric(gamma) ||
    length(gamma) != 1 ||
    !is.finite(gamma) ||
    gamma <= 0
  ) {

    stop("'gamma' must be a positive finite number or NULL.")

  }

}

# Validate degree
if (
  !is.numeric(degree) ||
  length(degree) != 1 ||
  !is.finite(degree) ||
  degree < 1 ||
  degree != floor(degree)
) {

  stop("'degree' must be a positive integer.")

}

# Validate coef0
if (
  !is.numeric(coef0) ||
  length(coef0) != 1 ||
  !is.finite(coef0)
) {

  stop("'coef0' must be a finite numeric value.")

}

# Validate scale
if (
  !is.logical(scale) ||
  length(scale) != 1 ||
  is.na(scale)
) {

  stop("'scale' must be TRUE or FALSE.")

}

# Validate predictors
valid_predictors <- vapply(
  X,
  function(x) is.numeric(x) || is.factor(x),
  logical(1)
)

if (any(!valid_predictors)) {

  stop(
    "Unsupported predictor types detected: ",
    paste(
      colnames(X)[!valid_predictors],
      collapse = ", "
    ),
    ". Only numeric and factor predictors are supported."
  )

}

# Validate numeric predictors
numeric_features <- names(X)[
  vapply(X, is.numeric, logical(1))
]

for (feature in numeric_features) {

  if (any(!is.finite(X[[feature]]))) {

    stop(
      "Predictor '", feature,
      "' contains non-finite values."
    )

  }

}

# Validate response
if (!is.factor(y)) {

  stop(
    "The target variable must be a factor for classification."
  )

}

if (nlevels(y) < 2) {

  stop(
    "At least two target classes are required."
  )

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
# 4. Preprocessing
# =============================================================================

preprocessing_results <- .fit_svm_preprocessor(
  x = X_train,
  scale = scale
)

preprocessor <- preprocessing_results$preprocessor

X_train_processed <- preprocessing_results$x

X_test_processed <- .apply_svm_preprocessor(
  x = X_test,
  preprocessor = preprocessor
)

gamma_user <- gamma

# Update effective gamma using processed dimensions
if (is.null(gamma_user)) {
  gamma <- 1 / ncol(X_train_processed)
}

# =============================================================================
# 5. Train SVM
# =============================================================================

svm_model <- e1071::svm(
  x = X_train_processed,
  y = y_train,
  type = "C-classification",
  kernel = kernel,
  cost = cost,
  gamma = gamma,
  degree = degree,
  coef0 = coef0,
  scale = FALSE,
  probability = TRUE
)

# =============================================================================
# 6. Validate levels and model family
# =============================================================================

# Store categorical predictor levels
factor_levels <- preprocessor$factor_levels

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
# 7. Predictions
# =============================================================================

pred_class_raw <- predict(
  svm_model,
  newdata = X_test_processed,
  probability = TRUE,
  decision.values = TRUE
)

pred_prob <- attr(
  pred_class_raw,
  "probabilities"
)

if (!is.null(pred_prob)) {

  pred_prob <- pred_prob[
    ,
    levels(y_train),
    drop = FALSE
  ]

}

decision_values <- attr(
  pred_class_raw,
  "decision.values"
)

pred_class <- factor(
  as.character(pred_class_raw),
  levels = levels(y_train)
)

# =============================================================================
# 8. Classification metrics
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
# 9. ROC / AUC
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
# 10. Build frozen model
# =============================================================================

frozen_model <- list(

  # Schema information
  schema_version = 1L,

  # Identification
  algorithm = "svm",
  model = svm_model,
  model_family = model_family,

  # Predictor information
  predictor_names = preprocessor$predictor_names,
  predictor_types = predictor_types,

  # Training data
  training_data = list(
    predictors = X_train,
    response = y_train
  ),

  preprocessing = list(
    method = "svm_preprocessor",
    preprocessor = preprocessor,
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
    method = "native_svm",
    threshold = NULL,
    probability = TRUE,
    decision_values = TRUE
  ),

  # Algorithm-specific parameters
  algorithm_parameters = list(
    kernel = kernel,
    cost = cost,
    gamma = gamma,
    gamma_user = gamma_user,
    degree = degree,
    coef0 = coef0,
    scale = scale
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
  "autolearn_svm",
  "list"
)

# =============================================================================
# 11. SVM diagnostics
# =============================================================================

svm_results <- .svm_diagnostics(
  model = svm_model,
  n_train = nrow(X_train_processed)
)

# =============================================================================
# 12. Permutation test
# =============================================================================

perm_results <- NULL
perm_acc <- NULL
perm_p <- NULL

if (permutation_test) {

  perm_results <- .svm_permutation_test(
    x_train = X_train_processed,
    y_train = y_train,
    x_test = X_test_processed,
    y_test = y_test,
    observed_accuracy = accuracy,
    kernel = kernel,
    cost = cost,
    gamma = gamma,
    degree = degree,
    coef0 = coef0,
    nperm = nperm,
    seed = seed
  )

  perm_acc <- perm_results$accuracy
  perm_p <- perm_results$p_value

}

# =============================================================================
# 13. Prediction data
# =============================================================================

prediction_data <- data.frame(
  Sample = rownames(X_test),
  Observed = y_test,
  Predicted = pred_class,
  stringsAsFactors = FALSE
)

# =============================================================================
# 14. Plots
# =============================================================================

plots <- list()
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
  # 14.2. ROC
  # ===============================================

  gg_roc <- NULL

  if (!is.null(roc_obj)) {

    gg_roc <- .plot_roc_curve(
      roc_object = roc_obj,
      auc_value = auc_val
    )

    plots$roc <- gg_roc

  }

  # ===============================================
  # 14.3. Decision boundary
  # ===============================================

  numeric_only <- all(
    vapply(X, is.numeric, logical(1))
  )

  if (
    numeric_only &&
    ncol(X) == 2 &&
    ncol(X_train_processed) == 2
  ) {

    gg_boundary <- .plot_svm_boundary(
      model = svm_model,
      x_train = X_train_processed,
      y_train = y_train,
      preprocessor = preprocessor,
      kernel = kernel
    )

    plots$decision_boundary <- gg_boundary

  }

  # ===============================================
  # 14.4. Permutation test
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
  # 14.5. Combined plot
  # ===============================================

  combined_plot <- .combine_classification_plots(
    plots = plots[
      intersect(
        names(plots),
        c("confusion", "roc", "decision_boundary")
      )
    ]
  )

  # ===============================================
  # 14.6. Print plots
  # ===============================================

  if (!is.null(plots$permutation)) {
    print(plots$permutation)
  }

  if (!is.null(combined_plot)) {
    print(combined_plot)
  }

}

# =============================================================================
# 15. Output
# =============================================================================

obj <- list(

  # Identification
  call = match.call(),
  algorithm = "svm",
  model_family = model_family,

  # Fitted models
  model = svm_model,
  frozen_model = frozen_model,

  # Data information
  features = colnames(X),
  train_samples = rownames(X_train),
  test_samples = rownames(X_test),

  # Predictions
  prediction_data = prediction_data,
  predictions = pred_class,
  probabilities = pred_prob,
  decision_values = decision_values,

  # Performance evaluation
  confusion_matrix = cm,
  confusion_table = cm_tab,
  metrics = metrics,
  roc = roc_obj,
  auc_ci = auc_ci,
  confidence_intervals = confidence_intervals,

  # Algorithm-specific results
  details = list(

    # Support vector diagnostics
    support_vectors = svm_results,

    # Preprocessing information
    preprocessing = preprocessor,

    # Permutation test
    permutation = if (permutation_test) {
      list(
        accuracy = perm_acc,
        observed_accuracy = accuracy,
        p_value = perm_p,
        nperm = nperm
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
    kernel = kernel,
    cost = cost,
    gamma = gamma,
    gamma_user = gamma_user,
    degree = degree,
    coef0 = coef0,
    scale = scale,
    seed = seed,
    positive_class = positive_class,
    permutation_test = permutation_test,
    nperm = if (permutation_test) nperm else NULL,
    plot = plot
  )

)

class(obj) <- "learn_svm"

print(obj)

return(invisible(obj))

}

# =============================================================================
# Print S3
# =============================================================================
#' Print method for learn_svm objects
#'
#' Displays a formatted summary of a Support Vector Machine classification
#' model.
#'
#' @param x An object of class \code{learn_svm}.
#' @param ... Further arguments (ignored).
#' @return Invisibly returns \code{x}.
#' @method print learn_svm
#' @export

print.learn_svm <- function(x, ...) {

  # Get confidence intervals
  get_ci <- function(metric) {
    .get_ci(
      confidence_intervals = x$confidence_intervals,
      metric = metric
    )
  }

.print_header("Support Vector Machine Results")

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
    "Target Variable:         ",
    x$parameters$target_var,
    " (", classes, ")\n",
    sep = ""
  )

  cat(
    "Model Family:           ",
    x$model_family,
    "\n"
  )

  cat(
    "Samples (Train/Test):   ",
    n_train,
    "/",
    n_test,
    "\n"
  )

  cat(
    "Features Available:     ",
    n_feat,
    "\n"
  )

  cat(
    "Kernel:                 ",
    x$parameters$kernel,
    "\n"
  )

  cat(
    "Cost:                   ",
    x$parameters$cost,
    "\n"
  )

  cat(
    "Gamma:                  ",
    signif(x$parameters$gamma, 5),
    "\n"
  )

  cat(
    "Scaling:                ",
    x$parameters$scale,
    "\n"
  )

})

# ===============================================
# Support vector diagnostics
# ===============================================

.print_block("Model Diagnostics", function() {

  cat(
    "Total Support Vectors:  ",
    x$details$support_vectors$total_support_vectors,
    "\n"
  )

  cat(
    "SV Proportion:          ",
    round(
      x$details$support_vectors$support_vector_proportion,
      4
    ),
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
