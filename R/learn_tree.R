# =============================================================================
# LEARN TREE
# =============================================================================

#' Decision Tree Classification
#'
#' Trains a Decision Tree classification model using a single data frame
#' containing a categorical target variable and one or more predictor variables.
#'
#' The function provides a general framework for classification tasks using
#' clinical, phenotypic, molecular, or other quantitative or categorical
#' features. Predictors can be selected explicitly using
#' \code{feature_subset}, allowing the same function to be used regardless of
#' the biological or clinical origin of the data.
#'
#' The function performs stratified train/test splitting, Decision Tree
#' training, test-set evaluation, and feature importance analysis. Optional
#' cost-complexity pruning can be applied to simplify the fitted tree.
#' Diagnostic plots can also be generated automatically.
#'
#' @param data Data frame containing the target variable and predictor
#'   variables. Each row represents a sample and each column represents a
#'   variable. The data frame may contain clinical, phenotypic, molecular, or
#'   other quantitative or categorical features.
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
#' @param cp Numeric. Complexity parameter controlling the initial growth of
#'   the Decision Tree (default: \code{0.01}). Splits that do not improve the
#'   model sufficiently according to this threshold are not retained during
#'   tree construction. A value of \code{0} allows the tree to grow without
#'   this initial complexity restriction, subject to the other tree
#'   constraints.
#'
#' @param minsplit Integer. Minimum number of observations required in a node
#'   before a split can be attempted (default: \code{20}).
#'
#' @param minbucket Optional integer. Minimum number of observations allowed
#'   in a terminal node. If \code{NULL}, the default value determined by
#'   \pkg{rpart} is used, calculated as \code{floor(minsplit / 3)}.
#'
#' @param maxdepth Integer. Maximum depth of the fitted tree (default:
#'   \code{30}). Must be between 1 and 30.
#'
#' @param seed Integer. Random seed used for reproducible data partitioning
#'   and model fitting (default: \code{123}).
#'
#' @param positive_class Optional character. Reference class used for binary
#'   ROC/AUC calculations. If \code{NULL}, the second level of the target
#'   factor is used.
#'
#' @param prune_tree Logical. Whether to apply cost-complexity pruning to the
#'   initially fitted tree (default: \code{FALSE}). When enabled, the pruning
#'   complexity parameter is selected using \code{prune_method}.
#'
#' @param prune_method Character. Criterion used to select the complexity
#'   parameter for pruning (default: \code{"1se"}). Available options are:
#'   \describe{
#'     \item{\code{"1se"}}{Selects the simplest tree whose cross-validated
#'       relative error is within one standard error of the minimum error.}
#'     \item{\code{"min"}}{Selects the tree associated with the minimum
#'       cross-validated relative error.}
#'   }
#'   Only used when \code{prune_tree = TRUE}.
#'
#' @param permutation_test Logical value indicating whether to perform a
#'   permutation test to evaluate model accuracy. Defaults to \code{FALSE}.
#'
#' @param n_perm Integer specifying the number of permutations used in the
#'   permutation test. Defaults to \code{1000}.
#'
#' @param plot Logical. Whether to generate and print the combined diagnostic
#'   plots and the Decision Tree visualization to the graphics device
#'   (default: \code{TRUE}).
#'
#' @return
#' Invisibly returns an object of class \code{learn_tree}, containing:
#' \describe{
#'   \item{call}{The original function call.}
#'   \item{algorithm}{Character string identifying the algorithm:
#'     \code{"decision_tree"}.}
#'   \item{model_family}{Classification type, either \code{"binomial"} or
#'     \code{"multinomial"}.}
#'   \item{model}{The final fitted Decision Tree model used for test-set
#'     predictions and evaluation. Corresponds to the original tree when
#'     pruning is disabled, or the pruned tree when pruning is enabled.}
#'   \item{frozen_model}{A model bundle containing the final fitted model
#'     and associated metadata, including predictor names, factor levels,
#'     target levels, and relevant model parameters.}
#'   \item{features}{Character vector containing the predictor variables
#'     used for model training.}
#'   \item{train_samples, test_samples}{Sample identifiers assigned to the
#'     training and test sets, when available.}
#'   \item{prediction_data}{Data frame containing sample identifiers,
#'     observed classes (\code{Observed}), and predicted classes
#'     (\code{Predicted}) for the test set.}
#'   \item{predictions}{Predicted class labels for the test set.}
#'   \item{probabilities}{Predicted class probabilities for the test set.}
#'   \item{confusion_matrix}{Confusion matrix summarizing observed and
#'     predicted classifications.}
#'   \item{confusion_table}{Tabular representation of the confusion matrix.}
#'   \item{metrics}{Named list containing classification performance
#'     metrics, including Accuracy, Kappa, Precision, Recall, Specificity,
#'     and F1-score. AUC is included for binary classification and is
#'     \code{NA} when not applicable.}
#'   \item{roc}{ROC object for binary classification. Returns
#'     \code{NULL} for multiclass classification.}
#'   \item{confidence_intervals}{A list containing confidence intervals for the
#'     classification performance metrics, estimated using non-parametric
#'     bootstrap resampling.}
#'   \item{details}{A list containing Decision Tree-specific results:
#'     \describe{
#'       \item{original_model}{The initially fitted, unpruned
#'         \code{rpart} model.}
#'       \item{pruned_model}{The pruned \code{rpart} model when pruning
#'         is enabled. Otherwise, \code{NULL}.}
#'       \item{importance}{Data frame containing variable importance
#'         scores derived from the fitted Decision Tree.}
#'       \item{top_features}{Data frame containing the highest-ranked
#'         predictors by variable importance, when available.}
#'       \item{importance_metric}{Name of the importance measure used.}
#'       \item{diagnostics}{List containing structural diagnostics of the
#'         final selected tree, including the number of splits, terminal
#'         nodes, maximum depth, and complexity parameter table.}
#'       \item{cptable}{Complexity parameter table associated with the
#'         final selected tree.}
#'       \item{pruning}{List containing pruning information, including
#'         whether pruning was applied, the selected method and complexity
#'         parameter, selection results, and structural diagnostics for
#'         the original and final trees.}
#'       \item{permutation}{Results of the optional permutation test, including
#'         observed accuracy and empirical p-value. Returns \code{NULL} when the
#'         test is not performed.}
#'       \item{comparison}{Data frame comparing the original and pruned
#'         models when pruning is enabled. Otherwise, \code{NULL}.}
#'     }}
#'   \item{plots}{List containing individual diagnostic plots, including
#'     the confusion matrix, feature importance, ROC curve, permutation test and
#'     Decision Tree plots when applicable. Empty when \code{plot = FALSE}.}
#'   \item{combined_plot}{Combined diagnostic plot. Returns \code{NULL}
#'     when \code{plot = FALSE}.}
#'   \item{parameters}{List containing the parameters used during model
#'     training and evaluation.}
#' }
#'
#' @details
#' \strong{Data structure:}
#'   The function expects a sample-by-variable data frame. The target variable
#'   and all predictors must be contained in the same object. This design keeps
#'   the Decision Tree implementation independent of the biological origin of
#'   the predictors. For example, age, categorical clinical variables, gene
#'   expression, protein abundance, or metabolite measurements can all be
#'   supplied through the same interface.
#'
#' \strong{Predictor selection:}
#'   If \code{feature_subset} is supplied, only the specified columns are used
#'   as predictors. If it is \code{NULL}, all columns other than
#'   \code{target_var} and \code{sample_col} are used. Predictors must be
#'   numeric or factors. This allows both quantitative and categorical
#'   variables to be included in the same model.
#'
#' \strong{Sample identifiers:}
#'   If \code{sample_col} is present, its values are used to track samples
#'   through the analysis and are excluded from the predictor matrix. Duplicate
#'   sample identifiers are not allowed.
#'
#' \strong{Stratification:}
#'   The train/test partition is performed using stratified sampling,
#'   preserving the class structure of the target variable across the two sets.
#'   At least two samples per class are required.
#'
#' \strong{Model training:}
#'   Decision Trees are fitted using \code{rpart} with
#'   \code{method = "class"}. The initial tree structure is controlled by
#'   \code{cp}, \code{minsplit}, \code{minbucket}, and \code{maxdepth}.
#'   These parameters regulate tree complexity and the minimum number of
#'   observations required for additional splits and terminal nodes.
#'
#' \strong{Cost-complexity pruning:}
#'   When \code{prune_tree = TRUE}, the initially fitted tree is simplified
#'   using cost-complexity pruning. The complexity parameter is selected from
#'   the tree's complexity parameter table using either the 1-SE criterion
#'   (\code{prune_method = "1se"}) or the minimum cross-validated relative
#'   error criterion (\code{prune_method = "min"}).
#'
#'   The 1-SE criterion favors a simpler tree whose cross-validated error is
#'   within one standard error of the minimum observed error. The minimum
#'   criterion instead selects the complexity parameter associated with the
#'   lowest cross-validated relative error.
#'
#'   The initial value of \code{cp} constrains tree growth before pruning.
#'   Consequently, pruning cannot recover splits that were not included in
#'   the initially fitted tree.
#'
#'   When pruning is enabled, both the original and pruned models are
#'   evaluated using the same held-out test set, allowing their predictive
#'   performance and structural complexity to be compared directly.
#'
#'   Classification performance metrics are accompanied by confidence intervals
#'   estimated using non-parametric bootstrap resampling of the held-out test
#'   samples. For binary classification, the AUC confidence interval is
#'   estimated using the DeLong method.
#'
#' \strong{Model evaluation:}
#'   Model performance is evaluated on the held-out test set using a
#'   confusion matrix, Accuracy, Kappa, and class-wise Precision, Recall,
#'   Specificity, and F1 score. Macro-averaged Precision, Recall, and F1
#'   are also reported. For binary classification problems, an ROC curve
#'   and AUC are calculated using the predicted class probabilities.
#'
#'   For multiclass classification, ROC analysis and AUC calculation are
#'   not performed. When pruning is enabled, the same evaluation metrics
#'   are calculated for both the original and pruned models, allowing
#'   their predictive performance and structural complexity to be compared.
#'   The comparison also includes the number of splits and terminal nodes
#'   in each tree.
#'
#' \strong{Variable importance:}
#'   Variable importance is obtained from the fitted Decision Tree using the
#'   importance information provided by \pkg{rpart}. The returned
#'   \code{top_features} object contains the ten highest-ranked predictors
#'   when variable importance scores are available. Importance scores reflect
#'   the contribution of predictors to the fitted tree and should not be
#'   interpreted as evidence of causal or mechanistic relationships.
#'
#' \strong{Visualization:}
#'   When \code{plot = TRUE}, the function generates a composite diagnostic
#'   visualization using \pkg{ggplot2} and \pkg{patchwork}. The combined
#'   visualization includes the confusion matrix, ROC curve for binary
#'   classification, and feature importance when available. A separate
#'   Decision Tree visualization is also generated using \pkg{rpart.plot}.
#'   Individual diagnostic plots are stored in the \code{plots} component,
#'   while the composite visualization is stored separately in
#'   \code{combined_plot}. When \code{plot = FALSE}, diagnostic plots are
#'   not generated.
#'
#' @examples
#' # Example: Train a Decision Tree using the iris dataset
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
#' tree_model <- learn_tree(
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
#' @importFrom stats predict complete.cases
#'
#' @export

learn_tree <- function(
    data,
    target_var,
    feature_subset = NULL,
    sample_col = "Sample",
    train_fraction = 0.7,
    cp = 0.01,
    minsplit = 20,
    minbucket = NULL,
    maxdepth = 30,
    seed = 123,
    positive_class = NULL,
    prune_tree = FALSE,
    prune_method = "1se",
    permutation_test = FALSE,
    n_perm = 1000,
    plot = TRUE
) {

# =============================================================================
# 0. Basic checks
# =============================================================================

.check_dependencies(c(
  "rpart",
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
  function(x) is.numeric(x) || is.factor(x),
  logical(1)
)

if (any(!supported)) {
  stop(
    "All predictors must be numeric or factors. ",
    "Unsupported variables detected: ",
    paste(
      colnames(X)[!supported],
      collapse = ", "
    )
  )
}

# Prune tree
if (!is.logical(prune_tree) ||
    length(prune_tree) != 1 ||
    is.na(prune_tree)) {
  stop("'prune_tree' must be TRUE or FALSE.")
}

# Prune method
if (!is.character(prune_method) ||
    length(prune_method) != 1 ||
    is.na(prune_method)) {
  stop("'prune_method' must be either '1se' or 'min'.")
}

prune_method <- match.arg(
  prune_method,
  choices = c("1se", "min")
)

# Identify predictor types
predictor_types <- .get_predictor_types(X)

# =============================================================================
# 3. Parameter validation
# =============================================================================

if (
  length(cp) != 1 ||
  !is.numeric(cp) ||
  !is.finite(cp) ||
  cp < 0
) {
  stop("cp must be a single non-negative numeric value.")
}

if (
  length(minsplit) != 1 ||
  !is.numeric(minsplit) ||
  !is.finite(minsplit) ||
  minsplit < 1 ||
  minsplit != as.integer(minsplit)
) {
  stop("minsplit must be a positive integer.")
}

if (!is.null(minbucket)) {
  if (
    length(minbucket) != 1 ||
    !is.numeric(minbucket) ||
    !is.finite(minbucket) ||
    minbucket < 1 ||
    minbucket != as.integer(minbucket)
  ) {
    stop("minbucket must be NULL or a positive integer.")
  }
}

if (
  length(maxdepth) != 1 ||
  !is.numeric(maxdepth) ||
  !is.finite(maxdepth) ||
  maxdepth < 1 ||
  maxdepth > 30 ||
  maxdepth != as.integer(maxdepth)
) {
  stop("maxdepth must be an integer between 1 and 30.")
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

n_classes <- nlevels(y_train)

if (n_classes < 2) {
  stop("The training set must contain at least two classes.")
}

model_family <- if (n_classes == 2) {
  "binomial"
} else {
  "multinomial"
}

if (model_family == "multinomial" && !is.null(positive_class)) {
  stop("'positive_class' is only applicable to binary classification.")
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
# 5. Train Decision Tree
# =============================================================================

train_data <- data.frame(
  .target = y_train,
  X_train,
  check.names = FALSE
)

tree_formula <- stats::reformulate(
  termlabels = colnames(X_train),
  response = ".target"
)

tree_control_args <- list(
  cp = cp,
  minsplit = minsplit,
  maxdepth = maxdepth
)

# Use base argument for minbucket when NULL
if (!is.null(minbucket)) {
  tree_control_args$minbucket <- minbucket
}

tree_control <- do.call(
  rpart::rpart.control,
  tree_control_args
)

tree_model <- rpart::rpart(
  formula = tree_formula,
  data = train_data,
  method = "class",
  control = tree_control,
  model = TRUE
)

# =============================================================================
# 6. Incorporate prune
# =============================================================================

# Initialize model selection objects
pruned_model <- NULL
cp_selection <- NULL
model_used <- tree_model

# Apply pruning when requested
if (prune_tree) {

  cp_selection <- .tree_select_cp(
    model = tree_model,
    method = prune_method
  )

  pruned_model <- rpart::prune(
    tree_model,
    cp = cp_selection$cp
  )

  model_used <- pruned_model
}

# =============================================================================
# 7. Model evaluation
# =============================================================================

# Evaluate original tree
original_eval <- .evaluate_tree_model(
  model = tree_model,
  X_test = X_test,
  y_test = y_test,
  positive_class = positive_class
)

# Evaluate selected model
if (prune_tree) {

  final_eval <- .evaluate_tree_model(
    model = model_used,
    X_test = X_test,
    y_test = y_test,
    positive_class = positive_class
  )

} else {

  final_eval <- original_eval

}

# Extract final predictions
pred_class <- final_eval$predictions
pred_prob <- final_eval$probabilities

# ===============================================
# 7.1. Classification metrics
# ===============================================

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

# ===============================================
# 7.2. ROC / AUC
# ===============================================

roc_obj <- NULL
auc_val <- NA_real_
auc_ci <- NULL
roc_results <- NULL

if (model_family == "binomial") {

  roc_results <- .classification_roc(
    observed = y_test,
    probabilities = pred_prob,
    positive_class = positive_class,
    conf_level = 0.95
  )

  positive_class <- roc_results$positive_class

  # Format DeLong confidence interval
  auc_ci_raw <- roc_results$auc_ci

  auc_ci <- list(
    lower = unname(auc_ci_raw[1]),
    estimate = unname(auc_ci_raw[2]),
    upper = unname(auc_ci_raw[3]),
    method = "delong"
  )

}

roc_obj <- roc_results$roc
auc_val <- roc_results$auc
metrics$auc <- auc_val

# ===============================================
# 7.3. Integrate AUC confidence interval
# ===============================================

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
# 8. Build prediction data
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

roc_obj <- if (!is.null(final_eval$roc)) {
  final_eval$roc$roc
} else {
  NULL
}

if (model_family == "binomial" && !is.null(final_eval$roc)) {
  positive_class <- final_eval$roc$positive_class
}

negative_class <- if (model_family == "binomial") {
  setdiff(levels(y), positive_class)
} else {
  NULL
}

# =============================================================================
# 9. Compare original and pruned models
# =============================================================================

comparison <- NULL

original_diag <- .tree_diagnostics(tree_model)
final_diag <- .tree_diagnostics(model_used)

if (prune_tree) {

  comparison <- data.frame(
    Model = c("Original", "Pruned"),

    Accuracy = c(
      original_eval$metrics$accuracy,
      final_eval$metrics$accuracy
    ),

    Kappa = c(
      original_eval$metrics$kappa,
      final_eval$metrics$kappa
    ),

    Macro_Precision = c(
      original_eval$metrics$macro_precision,
      final_eval$metrics$macro_precision
    ),

    Macro_Recall = c(
      original_eval$metrics$macro_recall,
      final_eval$metrics$macro_recall
    ),

    Macro_F1 = c(
      original_eval$metrics$macro_f1,
      final_eval$metrics$macro_f1
    ),

    AUC = c(
      if (!is.null(original_eval$roc)) {
        original_eval$roc$auc
      } else {
        NA_real_
      },
      if (!is.null(final_eval$roc)) {
        final_eval$roc$auc
      } else {
        NA_real_
      }
    ),

    Splits = c(
      original_diag$n_splits,
      final_diag$n_splits
    ),

    Leaves = c(
      original_diag$n_leaves,
      final_diag$n_leaves
    )
  )
}

# =============================================================================
# 10. Feature importance
# =============================================================================

importance_results <- .tree_feature_importance(
  model = model_used,
  top_n = 10
)

imp_df <- importance_results$importance
top_features <- importance_results$top_features
importance_col <- importance_results$importance_metric

# =============================================================================
# 11. Tree diagnostics
# =============================================================================

tree_diag <- final_diag

# =============================================================================
# 12. Permutation test
# =============================================================================

perm_acc <- NULL
perm_p <- NULL
perm_results <- NULL

if (permutation_test) {

  perm_results <- .tree_permutation_test(
    x_train = X_train,
    y_train = y_train,
    x_test = X_test,
    y_test = y_test,
    observed_accuracy = accuracy,
    cp = cp,
    minsplit = minsplit,
    minbucket = minbucket,
    maxdepth = maxdepth,
    prune_tree = prune_tree,
    prune_method = prune_method,
    n_perm = n_perm,
    seed = seed
  )

  perm_acc <- perm_results$accuracy
  perm_p <- perm_results$p_value
}

# =============================================================================
# 13. Plots
# =============================================================================

plots <- list()
combined_plot <- NULL
gg_roc <- NULL

if (plot) {

  # ===============================================
  # 13.1. Confusion matrix
  # ===============================================

  gg_cm <- .plot_confusion_matrix(
    confusion_table = cm_tab
  )

  plots$confusion <- gg_cm

  # ===============================================
  # 13.2. Feature importance
  # ===============================================

  gg_imp <- NULL

  if (nrow(top_features) > 0) {

    gg_imp <- .plot_feature_importance(
      feature_importance = top_features,
      importance_col = importance_col
    )

    plots$importance <- gg_imp
  }

  # ===============================================
  # 13.3. ROC
  # ===============================================

  if (!is.null(roc_obj)) {

    gg_roc <- .plot_roc_curve(
      roc_object = roc_obj,
      auc_value = auc_val
    )

    plots$roc <- gg_roc
  }

  # ===============================================
  # 13.4. Combined plot
  # ===============================================

  combined_plot <- .combine_classification_plots(
    plots = list(
      confusion = gg_cm,
      roc = gg_roc,
      importance = gg_imp,

      if (!is.null(plots$roc)) {
        spacer = patchwork::plot_spacer()
      }
    )
  )

  # ===============================================
  # 13.5. Decision tree
  # ===============================================

  plots$tree <- .plot_decision_tree(model_used)

  # ===============================================
  # 13.6. Permutation test
  # ===============================================

  if (!is.null(perm_acc)) {

    gg_perm <- .plot_permutation_test(
      permutation_accuracy = perm_acc,
      observed_accuracy = accuracy,
      p_value = perm_p
    )

    plots$permutation <- gg_perm

    if (!is.null(gg_perm)) {
      print(gg_perm)
    }
  }

  # ===============================================
  # 13.7. Print combined plots
  # ===============================================

  if (!is.null(combined_plot)) {
    print(combined_plot)
  }

}

# =============================================================================
# 14. Build frozen model
# =============================================================================

frozen_model <- list(

  # Schema information
  schema_version = 1L,

  # Identification
  algorithm = "tree",
  model = model_used,
  model_family = model_family,

  # Predictor information
  predictor_names = colnames(X),
  predictor_types = predictor_types,

  # Training data
  training_data = list(
    predictors = X_train,
    response = y_train
  ),

  preprocessing = list(
    method = "native_rpart",
    factor_levels = lapply(
      X_train[vapply(X_train, is.factor, logical(1))],
      levels
    )
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
    method = "maximum_probability",
    threshold = NULL
  ),

  # Algorithm-specific parameters
  algorithm_parameters = list(
    cp = cp,
    minsplit = minsplit,
    minbucket = minbucket,
    maxdepth = maxdepth,
    prune_tree = prune_tree,

    prune_method = if (prune_tree) {
      prune_method
    } else {
      NULL
    },

    selected_cp = if (prune_tree) {
      cp_selection$cp
    } else {
      NULL
    }
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

    evaluation_parameters = list(
      train_fraction = train_fraction,
      n_boot = 2000,
      conf_level = 0.95,
      bootstrap_seed = seed + 1
    )
  )
)

class(frozen_model) <- c(
  "autolearn_tree",
  "list"
)

# =============================================================================
# 15. Output
# =============================================================================

obj <- list(

  call = match.call(),

  # Common model information
  algorithm = "decision_tree",
  model_family = model_family,
  model = model_used,
  frozen_model = frozen_model,

  # Common data information
  features = colnames(X),
  train_samples = rownames(X_train),
  test_samples = rownames(X_test),

  # Common prediction information
  prediction_data = prediction_data,
  predictions = pred_class,
  probabilities = pred_prob,

  # Common evaluation information
  confusion_matrix = cm,
  confusion_table = cm_tab,
  metrics = metrics,
  roc = roc_obj,
  auc_ci = auc_ci,
  confidence_intervals = confidence_intervals,

  # Algorithm-specific information
  details = list(

    # Original and selected models
    original_model = tree_model,
    pruned_model = pruned_model,

    # Feature importance
    importance = imp_df,
    top_features = top_features,
    importance_metric = importance_col,

    # Tree diagnostics
    diagnostics = tree_diag,
    cptable = tree_diag$cptable,

    # Pruning information
    pruning = list(
      applied = prune_tree,
      method = if (prune_tree) prune_method else NULL,
      cp = if (prune_tree) cp_selection$cp else NULL,
      selection = cp_selection,
      original_diagnostics = original_diag,
      final_diagnostics = final_diag
    ),

    # Original versus pruned model comparison
    comparison = comparison,

    # Permutation test
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
    cp = cp,
    minsplit = minsplit,
    minbucket = minbucket,
    maxdepth = maxdepth,
    seed = seed,
    positive_class = positive_class,
    prune_tree = prune_tree,
    prune_method = prune_method,
    permutation_test = permutation_test,
    n_perm = n_perm,
    plot = plot
  )
)

class(obj) <- "learn_tree"

print(obj)

return(invisible(obj))

}

# =============================================================================
# Print S3
# =============================================================================
#' Print method for learn_tree objects
#'
#' Displays a formatted summary of a decision tree model.
#'
#' @param x An object of class \code{learn_tree}.
#' @param ... Further arguments (ignored).
#'
#' @return Invisibly returns \code{x}.
#' @method print learn_tree
#' @export

print.learn_tree <- function(x, ...) {

  # Get confidence intervals
  get_ci <- function(metric) {
    .get_ci(
      confidence_intervals = x$confidence_intervals,
      metric = metric
    )
  }

  .print_header("Decision Tree Results")

  n_train <- length(x$train_samples)
  n_test <- nrow(x$prediction_data)
  n_feat <- length(x$features)

  classes <- paste(
    x$frozen_model$response$target_levels,
    collapse = ", "
  )

  # ===========================================
  # Data & setup
  # ===========================================

  .print_block("Data & Setup", function() {

    cat(
      "Target Variable:       ",
      x$parameters$target_var,
      " (", classes, ")\n",
      sep = ""
    )

    cat("Samples (Train/Test):  ", n_train, "/", n_test, "\n")
    cat("Features Used:         ", n_feat, "\n")
    cat("Complexity Parameter:  ", x$parameters$cp, "\n")
    cat("Minimum Split:         ", x$parameters$minsplit, "\n")
    cat("Minimum Bucket:        ", x$parameters$minbucket, "\n")
    cat("Maximum Depth:         ", x$parameters$maxdepth, "\n")
  })

  # ===========================================
  # pruning parameters and results
  # ===========================================

  if (x$details$pruning$applied) {
   .print_block("Pruning", function() {

    cat("Pruning Method:        ", x$details$pruning$method, "\n")
    cat("Selected CP:           ", round(x$details$pruning$cp, 6), "\n")
    cat("Original Splits:       ",
        x$details$pruning$original_diagnostics$n_splits, "\n")
    cat("Final Splits:          ",
        x$details$pruning$final_diagnostics$n_splits, "\n")
    }
   )
  }

  # ===========================================
  # Tree structure
  # ===========================================

  .print_block("Tree Structure", function() {

    cat("Number of Splits:      ", x$details$diagnostics$n_splits, "\n")
    cat("Terminal Nodes:        ", x$details$diagnostics$n_leaves, "\n")
    cat("Maximum Depth Reached: ", x$details$diagnostics$max_depth, "\n")
  })

  # ===========================================
  # Performance
  # ===========================================

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
      "Macro Precision:       ",
      .format_metric_ci(
        x$metrics$macro_precision,
        get_ci("macro_precision")
      ),
      "\n"
    )

    cat(
      "Macro Recall:          ",
      .format_metric_ci(
        x$metrics$macro_recall,
        get_ci("macro_recall")
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
          get_ci("auc")
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

    if (
      is.data.frame(x$details$top_features) &&
      nrow(x$details$top_features) > 0
    ) {

      cat(
        "Top Feature:           ",
        as.character(x$details$top_features$Feature[1]),
        "\n"
      )

    } else {

      cat("Top Feature:            No splits available\n")

    }
  })

  # ===========================================
  # Model comparison
  # ===========================================

  if (!is.null(x$details$comparison)) {

    .print_block("Model Comparison", function() {

      comparison_display <- x$details$comparison

      numeric_cols <- vapply(
        comparison_display,
        is.numeric,
        logical(1)
      )

      comparison_display[numeric_cols] <- lapply(
        comparison_display[numeric_cols],
        round,
        digits = 4
      )

      print(
        comparison_display,
        row.names = FALSE
      )
    })
  }

  invisible(x)
}
