# =============================================================================
# UTILS SPECIFICS
# =============================================================================

# =============================================================================
# 1. Tree feature importance
# =============================================================================
#' @keywords internal

.tree_feature_importance <- function(model, top_n = 10) {

  # Validate top_n
  if (
    length(top_n) != 1 ||
    !is.numeric(top_n) ||
    !is.finite(top_n) ||
    top_n < 1 ||
    top_n != as.integer(top_n)
  ) {
    stop("top_n must be a positive integer.")
  }

  importance <- model$variable.importance

  # Handle trees without splits
  if (is.null(importance) || length(importance) == 0) {

    importance_df <- data.frame(
      Feature = character(),
      Importance = numeric()
    )

    return(list(
      importance = importance_df,
      top_features = importance_df,
      importance_metric = "Importance"
    ))
  }

  importance_df <- data.frame(
    Feature = names(importance),
    Importance = as.numeric(importance),
    row.names = NULL
  )

  importance_df <- importance_df[
    order(importance_df$Importance, decreasing = TRUE),
    ,
    drop = FALSE
  ]

  top_features <- head(
    importance_df,
    top_n
  )

  list(
    importance = importance_df,
    top_features = top_features,
    importance_metric = "Importance"
  )
}

# =============================================================================
# 2. Tree diagnosis
# =============================================================================
#' @keywords internal

.tree_diagnostics <- function(model) {

  frame <- model$frame

  if (is.null(frame) || nrow(frame) == 0) {
    stop("The decision tree model has an empty frame.")
  }

  is_leaf <- frame$var == "<leaf>"

  n_splits <- sum(!is_leaf)
  n_leaves <- sum(is_leaf)

  node_ids <- as.numeric(rownames(frame))

  max_depth <- if (length(node_ids) > 0) {
    max(floor(log2(node_ids)))
  } else {
    0
  }

  list(
    n_splits = n_splits,
    n_leaves = n_leaves,
    max_depth = max_depth,
    cptable = as.data.frame(model$cptable)
  )
}

# =============================================================================
# 3. tree prune
# =============================================================================
#' @keywords internal

.tree_select_cp <- function(model, method = "1se") {

  method <- match.arg(method, choices = c("1se", "min"))

  cptable <- as.data.frame(model$cptable)

  if (nrow(cptable) == 0) {
    stop("The model does not contain a valid complexity table.")
  }

  # Identify the row with the minimum cross-validation error
  min_idx <- which.min(cptable$xerror)

  if (method == "min") {

    selected_idx <- min_idx
    threshold <- cptable$xerror[min_idx]

  } else {

    # One-standard-error threshold
    threshold <- cptable$xerror[min_idx] +
      cptable$xstd[min_idx]

    eligible <- which(cptable$xerror <= threshold)

    # Select the simplest eligible tree
    selected_idx <- eligible[
      which.min(cptable$nsplit[eligible])
    ]
  }

  list(
    cp = cptable$CP[selected_idx],
    selected_row = selected_idx,
    min_row = min_idx,
    threshold = threshold,
    method = method,
    cptable = cptable
  )
}


# =============================================================================
# 4. Decision Tree permutation test
# =============================================================================
#' @keywords internal

.tree_permutation_test <- function(
    x_train,
    y_train,
    x_test,
    y_test,
    observed_accuracy,
    cp = 0.01,
    minsplit = 20,
    minbucket = NULL,
    maxdepth = 30,
    prune_tree = FALSE,
    prune_method = "1se",
    n_perm = 1000,
    seed = 123
) {

  # Validate training data
  if (nrow(x_train) != length(y_train)) {
    stop("x_train and y_train must have the same number of samples.")
  }

  if (nrow(x_test) != length(y_test)) {
    stop("x_test and y_test must have the same number of samples.")
  }

  if (ncol(x_train) != ncol(x_test)) {
    stop("Training and testing data must have the same number of features.")
  }

  if (!identical(colnames(x_train), colnames(x_test))) {
    stop("Training and testing features must have the same names and order.")
  }

  if (length(y_train) == 0 || length(y_test) == 0) {
    stop("Training and testing sets cannot be empty.")
  }

  if (anyNA(y_train) || anyNA(y_test)) {
    stop("Training and testing labels cannot contain missing values.")
  }

  if (!is.factor(y_train) || !is.factor(y_test)) {
    stop("y_train and y_test must be factors.")
  }

  # Validate observed accuracy
  if (
    length(observed_accuracy) != 1 ||
    !is.numeric(observed_accuracy) ||
    !is.finite(observed_accuracy) ||
    observed_accuracy < 0 ||
    observed_accuracy > 1
  ) {
    stop("observed_accuracy must be a single value between 0 and 1.")
  }

  # Validate number of permutations
  if (
    length(n_perm) != 1 ||
    !is.numeric(n_perm) ||
    !is.finite(n_perm) ||
    n_perm < 1 ||
    n_perm != as.integer(n_perm)
  ) {
    stop("n_perm must be a positive integer.")
  }

  # Validate pruning method
  prune_method <- match.arg(
    prune_method,
    choices = c("1se", "min")
  )

  # Build tree formula
  tree_formula <- stats::reformulate(
    termlabels = colnames(x_train),
    response = ".target"
  )

  # Build tree control
  tree_control_args <- list(
    cp = cp,
    minsplit = minsplit,
    maxdepth = maxdepth
  )

  if (!is.null(minbucket)) {
    tree_control_args$minbucket <- minbucket
  }

  tree_control <- do.call(
    rpart::rpart.control,
    tree_control_args
  )

  # Initialize permutation distribution
  perm_acc <- numeric(n_perm)

  set.seed(seed)

  for (i in seq_len(n_perm)) {

    # Permute training labels
    perm_y_train <- y_train[
      sample.int(length(y_train))
    ]

    # Prepare permuted training data
    perm_train_data <- data.frame(
      .target = perm_y_train,
      x_train,
      check.names = FALSE
    )

    # Train null model
    perm_model <- rpart::rpart(
      formula = tree_formula,
      data = perm_train_data,
      method = "class",
      control = tree_control,
      model = TRUE
    )

    # Apply the same pruning procedure
    if (prune_tree) {

      perm_cp_selection <- .tree_select_cp(
        model = perm_model,
        method = prune_method
      )

      perm_model <- rpart::prune(
        perm_model,
        cp = perm_cp_selection$cp
      )
    }

    # Predict test samples
    perm_pred <- predict(
      perm_model,
      newdata = x_test,
      type = "class"
    )

    # Calculate test accuracy
    perm_acc[i] <- mean(
      perm_pred == y_test
    )
  }

  # Empirical p-value
  perm_p <- (
    sum(perm_acc >= observed_accuracy) + 1
  ) / (
    n_perm + 1
  )

  list(
    accuracy = perm_acc,
    p_value = perm_p,
    null_mean = mean(perm_acc),
    null_sd = stats::sd(perm_acc)
  )
}

# =============================================================================
# 5. Evaluate tree model
# =============================================================================
#' @keywords internal

.evaluate_tree_model <- function(
    model,
    X_test,
    y_test,
    positive_class = NULL
) {

  # Class predictions
  pred_class <- predict(
    model,
    newdata = X_test,
    type = "class"
  )

  pred_class <- factor(
    as.character(pred_class),
    levels = levels(y_test)
  )

  # Class probabilities
  pred_prob <- predict(
    model,
    newdata = X_test,
    type = "prob"
  )

  # Classification metrics
  performance <- .classification_metrics(
    predicted = pred_class,
    observed = y_test
  )

  # ROC analysis
  roc_result <- NULL

  if (nlevels(y_test) == 2) {

    roc_result <- .classification_roc(
      observed = y_test,
      probabilities = pred_prob,
      positive_class = positive_class
    )

    performance$metrics$auc <- roc_result$auc
    performance$metrics$roc <- roc_result$roc
    performance$metrics$positive_class <- roc_result$positive_class
  }

  list(
    predictions = pred_class,
    probabilities = pred_prob,
    confusion_matrix = performance$confusion_matrix,
    confusion_table = performance$confusion_table,
    metrics = performance$metrics,
    roc = roc_result
  )
}

# =============================================================================
# 6. RF feature importance
# =============================================================================
#' @keywords internal

.rf_feature_importance <- function(
    model,
    top_n = 10
) {

  # Validate top_n
  if (
    length(top_n) != 1 ||
    !is.numeric(top_n) ||
    !is.finite(top_n) ||
    top_n < 1 ||
    top_n != as.integer(top_n)
  ) {
    stop("top_n must be a positive integer.")
  }

  # Extract importance
  imp <- randomForest::importance(model)

  # Convert to data.frame
  imp_df <- data.frame(
    Feature = rownames(imp),
    imp,
    check.names = FALSE,
    row.names = NULL
  )

  # Identify importance metric
  importance_candidates <- c(
    "MeanDecreaseAccuracy",
    "MeanDecreaseGini"
  )

  importance_col <- intersect(
    importance_candidates,
    colnames(imp_df)
  )

  if (length(importance_col) == 0) {
    stop("No recognized importance metric found.")
  }

  importance_col <- importance_col[1]

  # Order features
  imp_df <- imp_df[
    order(
      -imp_df[[importance_col]],
      na.last = TRUE
    ),
    ,
    drop = FALSE
  ]

  # Select top features
  top_features <- utils::head(
    imp_df,
    top_n
  )

  list(
    importance = imp_df,
    top_features = top_features,
    importance_metric = importance_col
  )
}

# =============================================================================
# 7. RF oob diagnostics
# =============================================================================
#' @keywords internal

.rf_oob_diagnostics <- function(model) {

  # Extract OOB error trajectory
  err_rate <- model$err.rate

  if (
    is.null(err_rate) ||
    !"OOB" %in% colnames(err_rate)
  ) {
    stop("OOB error information is unavailable.")
  }

  if (nrow(err_rate) == 0) {
    stop("OOB error trajectory is empty.")
  }

  oob_values <- err_rate[, "OOB"]

  # Final OOB error
  oob_error <- tail(
    oob_values,
    1
  )

  # Build trajectory data.frame
  oob_df <- data.frame(
    Tree = seq_along(oob_values),
    OOB = as.numeric(oob_values)
  )

  list(
    oob_error = as.numeric(oob_error),
    trajectory = oob_df
  )
}

# =============================================================================
# 8. RF permutation test
# =============================================================================
#' @keywords internal

.rf_permutation_test <- function(
    x_train,
    y_train,
    x_test,
    y_test,
    observed_accuracy,
    ntree = 500,
    mtry = NULL,
    nperm = 1000,
    seed = 123
) {

  # Validate ntree
  if (
    length(ntree) != 1 ||
    !is.numeric(ntree) ||
    !is.finite(ntree) ||
    ntree < 1 ||
    ntree != as.integer(ntree)
  ) {
    stop("ntree must be a positive integer.")
  }

  # Validate mtry
  if (!is.null(mtry)) {
    if (
      length(mtry) != 1 ||
      !is.numeric(mtry) ||
      !is.finite(mtry) ||
      mtry < 1 ||
      mtry != as.integer(mtry) ||
      mtry > ncol(x_train)
    ) {
      stop(
        "mtry must be NULL or a positive integer ",
        "not exceeding the number of features."
      )
    }
  }

  if (is.null(mtry)) {
    mtry <- floor(sqrt(ncol(x_train)))
  }

  # Validate training data
  if (nrow(x_train) != length(y_train)) {
    stop("x_train and y_train must have the same number of samples.")
  }

  if (nrow(x_test) != length(y_test)) {
    stop("x_test and y_test must have the same number of samples.")
  }

  if (ncol(x_train) != ncol(x_test)) {
    stop("Training and testing data must have the same number of features.")
  }

  if (!identical(colnames(x_train), colnames(x_test))) {
    stop("Training and testing features must have the same names and order.")
  }

  if (length(y_train) == 0 || length(y_test) == 0) {
    stop("Training and testing sets cannot be empty.")
  }

  if (anyNA(y_train) || anyNA(y_test)) {
    stop("Training and testing labels cannot contain missing values.")
  }

  # Validate observed accuracy
  if (
    length(observed_accuracy) != 1 ||
    !is.numeric(observed_accuracy) ||
    !is.finite(observed_accuracy) ||
    observed_accuracy < 0 ||
    observed_accuracy > 1
  ) {
    stop("observed_accuracy must be a single value between 0 and 1.")
  }

  # Validate number of permutations
  if (
    length(nperm) != 1 ||
    !is.numeric(nperm) ||
    !is.finite(nperm) ||
    nperm < 1 ||
    nperm != as.integer(nperm)
  ) {
    stop("nperm must be a positive integer.")
  }

  # Initialize permutation distribution
  perm_acc <- numeric(nperm)

  set.seed(seed)

  for (i in seq_len(nperm)) {

    # Permute training labels
    perm_y_train <- sample(y_train)

    # Train null model
    perm_model <- randomForest::randomForest(
      x = x_train,
      y = perm_y_train,
      ntree = ntree,
      mtry = mtry
    )

    # Predict test samples
    perm_pred <- predict(
      perm_model,
      newdata = x_test
    )

    # Calculate test accuracy
    perm_acc[i] <- mean(
      perm_pred == y_test
    )
  }

  # Empirical p-value
  perm_p <- (
    sum(perm_acc >= observed_accuracy) + 1
  ) / (
    nperm + 1
  )

  list(
    accuracy = perm_acc,
    p_value = perm_p,
    null_mean = mean(perm_acc),
    null_sd = stats::sd(perm_acc)
  )
}

# =============================================================================
# 9. SVM preprocessor: fit
# =============================================================================
#' @keywords internal

.fit_svm_preprocessor <- function(
    x,
    scale = TRUE
) {

  # Validate predictor types
  valid <- vapply(
    x,
    function(z) is.numeric(z) || is.factor(z),
    logical(1)
  )

  if (any(!valid)) {
    stop(
      "SVM supports only numeric and factor predictors. ",
      "Invalid predictors: ",
      paste(names(x)[!valid], collapse = ", ")
    )
  }

  # Check numeric variables
  numeric_features <- names(x)[
    vapply(x, is.numeric, logical(1))
  ]

  for (feature in numeric_features) {

    if (any(!is.finite(x[[feature]]))) {
      stop(
        "Predictor '", feature,
        "' contains non-finite values."
      )
    }
  }

  # Store observed factor levels in training data
  factor_features <- names(x)[
    vapply(x, is.factor, logical(1))
  ]

  factor_levels <- list()

  for (feature in factor_features) {

    x[[feature]] <- droplevels(x[[feature]])

    if (nlevels(x[[feature]]) < 2) {
      stop(
        "Factor predictor '", feature,
        "' has fewer than two observed levels in training data."
      )
    }

    factor_levels[[feature]] <- levels(x[[feature]])
  }

  # Build model matrix
  terms_obj <- stats::terms(~ . - 1, data = x)

  model_frame <- stats::model.frame(
    terms_obj,
    data = x,
    na.action = stats::na.fail
  )

  x_matrix <- stats::model.matrix(
    terms_obj,
    data = model_frame
  )

  # Remove zero-variance predictors
  feature_sd <- apply(
    x_matrix,
    2,
    stats::sd
  )

  keep <- is.finite(feature_sd) & feature_sd > 0

  if (!any(keep)) {
    stop(
      "No variable with non-zero variance remains after encoding."
    )
  }

  dropped_features <- colnames(x_matrix)[!keep]

  x_matrix <- x_matrix[, keep, drop = FALSE]

  # Learn scaling parameters
  centers <- colMeans(x_matrix)
  scales <- apply(x_matrix, 2, stats::sd)

  if (scale) {

    x_matrix <- sweep(
      x_matrix,
      2,
      centers,
      "-"
    )

    x_matrix <- sweep(
      x_matrix,
      2,
      scales,
      "/"
    )

  } else {

    centers[] <- 0
    scales[] <- 1

  }

  # Store preprocessing information
  preprocessor <- list(
    terms = terms_obj,
    factor_levels = factor_levels,
    feature_names = colnames(x_matrix),
    predictor_names = names(x),
    dropped_features = dropped_features,
    centers = centers,
    scales = scales,
    scale = scale
  )

  list(
    x = x_matrix,
    preprocessor = preprocessor
  )

}

# =============================================================================
# 10. SVM preprocessor: apply
# =============================================================================
#' @keywords internal

.apply_svm_preprocessor <- function(
    x,
    preprocessor
) {

  # Validate required predictors
  required_features <- preprocessor$predictor_names

  missing_features <- setdiff(
    required_features,
    colnames(x)
  )

  if (length(missing_features) > 0) {
    stop(
      "Missing predictors: ",
      paste(missing_features, collapse = ", ")
    )
  }

  # Keep original predictor order
  x <- x[, required_features, drop = FALSE]

  # Validate and restore factor levels
  for (feature in names(preprocessor$factor_levels)) {

    observed <- as.character(x[[feature]])

    unseen <- setdiff(
      unique(observed[!is.na(observed)]),
      preprocessor$factor_levels[[feature]]
    )

    if (length(unseen) > 0) {
      stop(
        "New unseen levels detected in predictor '",
        feature, "': ",
        paste(unseen, collapse = ", ")
      )
    }

    x[[feature]] <- factor(
      observed,
      levels = preprocessor$factor_levels[[feature]]
    )

  }

  # Validate numeric variables
  numeric_features <- names(x)[
    vapply(x, is.numeric, logical(1))
  ]

  for (feature in numeric_features) {

    if (any(!is.finite(x[[feature]]))) {
      stop(
        "Predictor '", feature,
        "' contains non-finite values."
      )
    }

  }

  # Reconstruct model matrix
  model_frame <- stats::model.frame(
    preprocessor$terms,
    data = x,
    xlev = preprocessor$factor_levels,
    na.action = stats::na.fail
  )

  x_matrix <- stats::model.matrix(
    preprocessor$terms,
    data = model_frame
  )

  # Keep exactly the training features
  x_matrix <- x_matrix[
    ,
    preprocessor$feature_names,
    drop = FALSE
  ]

  # Apply training transformations
  x_matrix <- sweep(
    x_matrix,
    2,
    preprocessor$centers,
    "-"
  )

  x_matrix <- sweep(
    x_matrix,
    2,
    preprocessor$scales,
    "/"
  )

  x_matrix

}

# =============================================================================
# 11. SVM diagnostics
# =============================================================================
#' @keywords internal

.svm_diagnostics <- function(
    model,
    n_train
) {

  # Total support vectors
  total_sv <- nrow(model$SV)

  # Support vector proportion
  sv_proportion <- total_sv / n_train

  # Return diagnostics
  list(
    total_support_vectors = total_sv,
    support_vector_proportion = sv_proportion,
    support_vectors_per_class = model$nSV,
    support_vector_indices = model$index
  )

}

# =============================================================================
# 12. SVM permutation test
# =============================================================================
#' @keywords internal

.svm_permutation_test <- function(
    x_train,
    y_train,
    x_test,
    y_test,
    observed_accuracy,
    kernel,
    cost,
    gamma,
    degree,
    coef0,
    nperm = 1000,
    seed = 123
) {

  # Validate number of permutations
  if (
    !is.numeric(nperm) ||
    length(nperm) != 1 ||
    !is.finite(nperm) ||
    nperm < 1 ||
    nperm != floor(nperm)
  ) {

    stop("'nperm' must be a positive integer.")

  }

  # Reproducibility
  set.seed(seed)

  # Initialize permutation accuracy
  perm_accuracy <- numeric(nperm)

  # Permutation loop
  for (i in seq_len(nperm)) {

    # Shuffle training labels
    y_perm <- sample(y_train)

    # Fit permuted SVM
    perm_model <- e1071::svm(
      x = x_train,
      y = y_perm,
      type = "C-classification",
      kernel = kernel,
      cost = cost,
      gamma = gamma,
      degree = degree,
      coef0 = coef0,
      scale = FALSE,
      probability = FALSE
    )

    # Predict test samples
    perm_pred <- predict(
      perm_model,
      newdata = x_test
    )

    # Calculate accuracy
    perm_accuracy[i] <- mean(
      as.character(perm_pred) ==
        as.character(y_test)
    )

  }

  # Empirical permutation p-value
  p_value <- (
    1 + sum(
      perm_accuracy >= observed_accuracy
    )
  ) / (nperm + 1)

  # Return results
  list(
    accuracy = perm_accuracy,
    observed_accuracy = observed_accuracy,
    p_value = p_value,
    nperm = nperm
  )

}

# =============================================================================
# 13. Validate compare input
# =============================================================================
#' @keywords internal

.validate_compare_input <- function(results) {

  # Validate input
  if (!is.list(results) || length(results) == 0) {
    stop("'results' must be a non-empty list.")
  }

  # Validate names
  if (
    is.null(names(results)) ||
    anyNA(names(results)) ||
    any(names(results) == "") ||
    anyDuplicated(names(results))
  ) {
    stop(
      "All elements in 'results' must have unique, non-empty names."
    )
  }

  # Validate supported classes
  supported_classes <- c(
    "learn_tree",
    "learn_logistic",
    "learn_elastic_net",
    "learn_random_forest",
    "learn_svm"
  )

  valid_objects <- vapply(
    results,
    function(x) {
      any(vapply(
        supported_classes,
        function(cl) inherits(x, cl),
        logical(1)
      ))
    },
    logical(1)
  )

  if (!all(valid_objects)) {
    invalid <- names(results)[!valid_objects]

    stop(
      "Unsupported object(s) in 'results': ",
      paste(invalid, collapse = ", "),
      ". All elements must be objects returned by supported learn_* functions."
    )
  }

  # Validate required fields
  required_fields <- c(
    "algorithm",
    "prediction_data",
    "train_samples",
    "test_samples",
    "parameters"
  )

  valid_structure <- vapply(
    results,
    function(x) {
      all(required_fields %in% names(x))
    },
    logical(1)
  )

  if (!all(valid_structure)) {
    invalid <- names(results)[!valid_structure]

    stop(
      "Object(s) with incomplete structure: ",
      paste(invalid, collapse = ", ")
    )
  }

  invisible(TRUE)
}

# =============================================================================
# 13. Check compare compatibility
# =============================================================================
#' @keywords internal

.check_compare_compatibility <- function(results) {

  # Extract reference object
  reference <- results[[1]]

  reference_target <- reference$parameters$target_var
  reference_classes <- levels(
    reference$prediction_data$Observed
  )
  reference_test <- reference$test_samples
  reference_train <- reference$train_samples
  reference_positive <- reference$parameters$positive_class

  # Initialize compatibility checks
  checks <- data.frame(
    check = character(),
    status = character(),
    message = character(),
    stringsAsFactors = FALSE
  )

  add_check <- function(name, status, message) {
    checks <<- rbind(
      checks,
      data.frame(
        check = name,
        status = status,
        message = message,
        stringsAsFactors = FALSE
      )
    )
  }

  # Check target variable
  target_match <- vapply(
    results,
    function(x) {
      identical(x$parameters$target_var, reference_target)
    },
    logical(1)
  )

  if (all(target_match)) {
    add_check(
      "target_var",
      "compatible",
      "All models use the same target variable."
    )
  } else {
    add_check(
      "target_var",
      "incompatible",
      "Models use different target variables."
    )
  }

  # Check class levels
  class_match <- vapply(
    results,
    function(x) {
      identical(
        levels(x$prediction_data$Observed),
        reference_classes
      )
    },
    logical(1)
  )

  if (all(class_match)) {
    add_check(
      "class_levels",
      "compatible",
      "All models have identical class levels and ordering."
    )
  } else {
    add_check(
      "class_levels",
      "incompatible",
      "Models have different class levels or ordering."
    )
  }

  # Check test samples
  test_match <- vapply(
    results,
    function(x) {
      setequal(x$test_samples, reference_test)
    },
    logical(1)
  )

  if (all(test_match)) {
    add_check(
      "test_samples",
      "compatible",
      "All models were evaluated on the same test samples."
    )
  } else {
    add_check(
      "test_samples",
      "warning",
      "Models were evaluated on different test samples."
    )
  }

  # Check observed labels
  observed_match <- vapply(
    results,
    function(x) {

      if (!setequal(
        x$prediction_data$Sample,
        reference$prediction_data$Sample
      )) {
        return(FALSE)
      }

      reference_data <- reference$prediction_data
      current_data <- x$prediction_data

      reference_order <- match(
        reference_data$Sample,
        current_data$Sample
      )

      identical(
        as.character(reference_data$Observed),
        as.character(current_data$Observed[reference_order])
      )
    },
    logical(1)
  )

  if (all(observed_match)) {
    add_check(
      "observed_labels",
      "compatible",
      "Observed labels are identical for corresponding samples."
    )
  } else {
    add_check(
      "observed_labels",
      "incompatible",
      "Observed labels differ between models for corresponding samples."
    )
  }

  # Check training samples
  train_match <- vapply(
    results,
    function(x) {
      setequal(x$train_samples, reference_train)
    },
    logical(1)
  )

  if (all(train_match)) {
    add_check(
      "train_samples",
      "compatible",
      "All models were trained using the same samples."
    )
  } else {
    add_check(
      "train_samples",
      "warning",
      "Models were trained using different samples."
    )
  }

  # Check positive class
  positive_match <- vapply(
    results,
    function(x) {
      identical(
        x$parameters$positive_class,
        reference_positive
      )
    },
    logical(1)
  )

  if (all(positive_match)) {
    add_check(
      "positive_class",
      "compatible",
      "All models use the same positive class."
    )
  } else {
    add_check(
      "positive_class",
      "warning",
      "Models use different positive classes."
    )
  }

  # Determine general compatibility
  comparable <- !any(
    checks$status == "incompatible"
  )

  # Determine ROC compatibility
  roc_compatible <- (
    comparable &&
      all(positive_match) &&
      length(reference_classes) == 2 &&
      !is.null(reference_positive) &&
      length(reference_positive) == 1 &&
      reference_positive %in% reference_classes
  )

  # Return compatibility report
  list(
    checks = checks,
    comparable = comparable,
    roc_compatible = roc_compatible
  )
}

# =============================================================================
# 14. Extract compare data
# =============================================================================
#' @keywords internal

.extract_compare_data <- function(results) {

  # Initialize output lists
  predictions_list <- list()
  probabilities_list <- list()
  confidence_intervals_list <- list()

  # Extract data from each model
  for (i in seq_along(results)) {

    model <- results[[i]]
    algorithm_name <- names(results)[i]

    prediction_data <- model$prediction_data

    # Extract predictions
    predictions_list[[i]] <- data.frame(
      Sample = prediction_data$Sample,
      Observed = prediction_data$Observed,
      Predicted = prediction_data$Predicted,
      Algorithm = algorithm_name,
      stringsAsFactors = FALSE
    )

    # Extract confidence intervals
    confidence_intervals <- model$confidence_intervals

    if (
      is.null(confidence_intervals) ||
      !is.data.frame(confidence_intervals)
    ) {
      stop(
        "Confidence intervals are unavailable for model: ",
        algorithm_name
      )
    }

    confidence_intervals$Algorithm <- algorithm_name

    confidence_intervals_list[[i]] <- confidence_intervals

    # Extract probabilities
    probabilities <- model$probabilities

    if (is.null(probabilities)) {
      next
    }

    # Validate probability matrix
    if (!is.matrix(probabilities) ||
        !is.numeric(probabilities)) {
      stop(
        "Invalid probability matrix in model: ",
        algorithm_name
      )
    }

    if (is.null(rownames(probabilities)) ||
        anyDuplicated(rownames(probabilities))) {
      stop(
        "Probability matrix must have unique sample identifiers: ",
        algorithm_name
      )
    }

    if (anyDuplicated(colnames(probabilities))) {
      stop(
        "Probability matrix contains duplicated class names: ",
        algorithm_name
      )
    }

    if (!setequal(
      rownames(probabilities),
      prediction_data$Sample
    )) {
      stop(
        "Probability matrix samples do not match prediction_data: ",
        algorithm_name
      )
    }

    # Align probabilities with prediction data
    sample_order <- match(
      prediction_data$Sample,
      rownames(probabilities)
    )

    probabilities <- probabilities[
      sample_order,
      ,
      drop = FALSE
    ]

    # Validate probability values
    if (anyNA(probabilities) ||
        any(!is.finite(probabilities)) ||
        any(probabilities < 0 | probabilities > 1)) {
      stop(
        "Invalid probability values in model: ",
        algorithm_name
      )
    }

    # Validate probability column names
    if (!setequal(
      colnames(probabilities),
      levels(prediction_data$Observed)
    )) {
      stop(
        "Probability columns do not match observed classes: ",
        algorithm_name
      )
    }

    # Store probabilities
    probabilities_list[[i]] <- data.frame(
      Sample = prediction_data$Sample,
      Algorithm = algorithm_name,
      probabilities,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  }

  # Combine predictions
  predictions <- do.call(
    rbind,
    predictions_list
  )

  rownames(predictions) <- NULL

  # Combine confidence intervals
  confidence_intervals <- do.call(
    rbind,
    confidence_intervals_list
  )

  rownames(confidence_intervals) <- NULL

  # Combine probabilities
  probabilities_list <- Filter(
    Negate(is.null),
    probabilities_list
  )

  if (length(probabilities_list) > 0) {

    probabilities <- do.call(
      rbind,
      probabilities_list
    )

    rownames(probabilities) <- NULL

  } else {

    probabilities <- NULL

  }

  # Return extracted data
  list(
    predictions = predictions,
    probabilities = probabilities,
    confidence_intervals = confidence_intervals
  )
}

# =============================================================================
# 15. Calculate compare metrics
# =============================================================================
#' @keywords internal

.calculate_compare_metrics <- function(
    compare_data,
    average = "macro"
) {

  # Validate average method
  average <- match.arg(
    average,
    choices = c("macro", "weighted")
  )

  predictions <- compare_data$predictions

  if (is.null(predictions) || nrow(predictions) == 0) {
    stop("'compare_data$predictions' cannot be empty.")
  }

  required_columns <- c(
    "Sample",
    "Observed",
    "Predicted",
    "Algorithm"
  )

  if (!all(required_columns %in% names(predictions))) {
    stop("Invalid prediction data structure.")
  }

  # Get algorithms and class levels
  algorithms <- unique(predictions$Algorithm)
  class_levels <- levels(predictions$Observed)

  if (is.null(class_levels) || length(class_levels) < 2) {
    stop("At least two classification levels are required.")
  }

  # Helper for safe metric aggregation
  .aggregate_metric <- function(values, weights) {

    valid <- !is.na(values)

    if (!any(valid)) {
      return(NA_real_)
    }

    if (average == "macro") {
      return(mean(values[valid]))
    }

    weighted.mean(
      values[valid],
      weights[valid]
    )
  }

  # Initialize output
  classwise_list <- vector("list", length(algorithms))
  summary_list <- vector("list", length(algorithms))

  # Calculate metrics for each algorithm
  for (i in seq_along(algorithms)) {

    algorithm <- algorithms[i]

    current <- predictions[
      predictions$Algorithm == algorithm,
      ,
      drop = FALSE
    ]

    observed <- factor(
      current$Observed,
      levels = class_levels
    )

    predicted <- factor(
      current$Predicted,
      levels = class_levels
    )

    if (anyNA(observed) || anyNA(predicted)) {
      stop(
        "Missing or invalid class labels in algorithm: ",
        algorithm
      )
    }

    # Confusion matrix
    cm <- table(
      Observed = observed,
      Predicted = predicted
    )

    total <- sum(cm)

    # Per-class metrics
    tp <- diag(cm)
    fn <- rowSums(cm) - tp
    fp <- colSums(cm) - tp
    tn <- total - tp - fn - fp

    support <- rowSums(cm)

    precision <- ifelse(
      tp + fp > 0,
      tp / (tp + fp),
      NA_real_
    )

    recall <- ifelse(
      tp + fn > 0,
      tp / (tp + fn),
      NA_real_
    )

    specificity <- ifelse(
      tn + fp > 0,
      tn / (tn + fp),
      NA_real_
    )

    f1 <- ifelse(
      2 * tp + fp + fn > 0,
      2 * tp / (2 * tp + fp + fn),
      NA_real_
    )

    classwise <- data.frame(
      Algorithm = algorithm,
      Class = class_levels,
      Support = as.numeric(support),
      Precision = as.numeric(precision),
      Recall = as.numeric(recall),
      Specificity = as.numeric(specificity),
      F1 = as.numeric(f1),
      stringsAsFactors = FALSE
    )

    classwise_list[[i]] <- classwise

    # Global metrics
    accuracy <- sum(tp) / total

    expected_agreement <- sum(
      rowSums(cm) * colSums(cm)
    ) / total^2

    kappa <- if (
      isTRUE(all.equal(expected_agreement, 1))
    ) {
      NA_real_
    } else {
      (accuracy - expected_agreement) /
        (1 - expected_agreement)
    }

    # Aggregated metrics
    summary_list[[i]] <- data.frame(
      Algorithm = algorithm,
      Accuracy = accuracy,
      Kappa = kappa,
      Balanced_Accuracy = .aggregate_metric(
        recall,
        support
      ),
      Precision = .aggregate_metric(
        precision,
        support
      ),
      Recall = .aggregate_metric(
        recall,
        support
      ),
      Specificity = .aggregate_metric(
        specificity,
        support
      ),
      F1 = .aggregate_metric(
        f1,
        support
      ),
      stringsAsFactors = FALSE
    )

    # Balanced accuracy is always macro-averaged recall
    summary_list[[i]]$Balanced_Accuracy <- mean(
      recall,
      na.rm = TRUE
    )

    if (is.nan(summary_list[[i]]$Balanced_Accuracy)) {
      summary_list[[i]]$Balanced_Accuracy <- NA_real_
    }
  }

  # Combine results
  classwise <- do.call(rbind, classwise_list)
  summary <- do.call(rbind, summary_list)

  rownames(classwise) <- NULL
  rownames(summary) <- NULL

  # Return metrics
  list(
    summary = summary,
    classwise = classwise,
    average = average
  )
}

# =============================================================================
# 16. Calculate compare ROC
# =============================================================================
#' @keywords internal

.calculate_compare_roc <- function(
    compare_data,
    positive_class
) {

  # Validate input structure
  if (
    is.null(compare_data$predictions) ||
    is.null(compare_data$probabilities)
  ) {
    stop(
      "'compare_data' must contain both predictions and probabilities."
    )
  }

  predictions <- compare_data$predictions
  probabilities <- compare_data$probabilities

  required_prediction_cols <- c(
    "Sample", "Observed", "Predicted", "Algorithm"
  )

  if (!all(required_prediction_cols %in% names(predictions))) {
    stop(
      "'predictions' is missing required columns."
    )
  }

  if (!all(c("Sample", "Algorithm") %in% names(probabilities))) {
    stop(
      "'probabilities' is missing required columns."
    )
  }

  # Validate binary classification
  class_levels <- levels(predictions$Observed)

  if (
    is.null(class_levels) ||
    length(class_levels) != 2
  ) {
    stop(
      "ROC analysis is only available for binary classification."
    )
  }

  if (
    length(positive_class) != 1 ||
    is.na(positive_class) ||
    !positive_class %in% class_levels
  ) {
    stop(
      "'positive_class' must match one of the observed class levels."
    )
  }

  negative_class <- setdiff(
    class_levels,
    positive_class
  )

  # Identify algorithms
  algorithms <- unique(predictions$Algorithm)

  if (!setequal(
    algorithms,
    unique(probabilities$Algorithm)
  )) {
    stop(
      "All algorithms must have probability estimates for ROC comparison."
    )
  }

  # Initialize output lists
  curves_list <- vector("list", length(algorithms))
  summary_list <- vector("list", length(algorithms))

  # Initialize output lists
  curves_list <- vector("list", length(algorithms))
  summary_list <- vector("list", length(algorithms))
  roc_objects <- vector("list", length(algorithms))

  names(roc_objects) <- algorithms

  # Calculate ROC for each algorithm
  for (i in seq_along(algorithms)) {

    algorithm_name <- algorithms[i]

    algorithm_predictions <- predictions[
      predictions$Algorithm == algorithm_name,
      ,
      drop = FALSE
    ]

    algorithm_probabilities <- probabilities[
      probabilities$Algorithm == algorithm_name,
      ,
      drop = FALSE
    ]

    # Validate sample identifiers
    if (
      anyDuplicated(algorithm_predictions$Sample) ||
      anyDuplicated(algorithm_probabilities$Sample) ||
      !setequal(
        algorithm_predictions$Sample,
        algorithm_probabilities$Sample
      )
    ) {
      stop(
        "Prediction and probability samples do not match for algorithm: ",
        algorithm_name
      )
    }

    # Align probabilities with predictions
    sample_order <- match(
      algorithm_predictions$Sample,
      algorithm_probabilities$Sample
    )

    algorithm_probabilities <- algorithm_probabilities[
      sample_order,
      ,
      drop = FALSE
    ]

    # Validate positive class probability
    if (!positive_class %in% names(algorithm_probabilities)) {
      stop(
        "Positive class probability is missing for algorithm: ",
        algorithm_name
      )
    }

    scores <- algorithm_probabilities[[positive_class]]

    if (
      !is.numeric(scores) ||
      anyNA(scores) ||
      any(!is.finite(scores)) ||
      any(scores < 0 | scores > 1)
    ) {
      stop(
        "Invalid probability estimates for algorithm: ",
        algorithm_name
      )
    }

    # Prepare observed response
    observed <- factor(
      as.character(algorithm_predictions$Observed),
      levels = c(negative_class, positive_class)
    )

    if (
      anyNA(observed) ||
      length(unique(observed)) != 2
    ) {
      stop(
        "Both classes must be represented in the test set for algorithm: ",
        algorithm_name
      )
    }

    # Calculate ROC
    roc_object <- pROC::roc(
      response = observed,
      predictor = scores,
      levels = c(negative_class, positive_class),
      direction = "<",
      quiet = TRUE
    )

    # Store ROC object
    roc_objects[[i]] <- roc_object

    # Extract ROC coordinates
    roc_coordinates <- as.data.frame(
      pROC::coords(
        roc_object,
        x = "all",
        ret = c(
          "threshold",
          "specificity",
          "sensitivity"
        ),
        transpose = FALSE
      )
    )

    auc_value <- as.numeric(
      pROC::auc(roc_object)
    )

    # Store curve data
    curves_list[[i]] <- data.frame(
      Algorithm = algorithm_name,
      Threshold = roc_coordinates$threshold,
      Specificity = roc_coordinates$specificity,
      Sensitivity = roc_coordinates$sensitivity,
      FPR = 1 - roc_coordinates$specificity,
      TPR = roc_coordinates$sensitivity
    )

    # Store AUC summary
    summary_list[[i]] <- data.frame(
      Algorithm = algorithm_name,
      AUC = auc_value
    )
  }

  # Combine results
  curves <- do.call(rbind, curves_list)
  summary <- do.call(rbind, summary_list)

  rownames(curves) <- NULL
  rownames(summary) <- NULL

  # Return ROC results
  list(
    summary = summary,
    curves = curves,
    roc_objects = roc_objects
  )
}

# =============================================================================
# 17. Build prediction comparison summary
# =============================================================================
#' @keywords internal

.build_predict_comparison_summary <- function(
    metrics_comparison,
    auc_comparison = NULL
) {

  if (
    is.null(metrics_comparison$comparison) ||
    !is.data.frame(metrics_comparison$comparison)
  ) {
    stop(
      "'metrics_comparison' must contain a valid comparison table."
    )
  }

  summary <- metrics_comparison$comparison

  # Remove redundant specificity metrics
  summary <- summary[
    summary$Metric != "specificity",
    ,
    drop = FALSE
  ]

  # Add AUC comparison
  if (!is.null(auc_comparison)) {

    auc_data <- auc_comparison$comparison

    auc_row <- data.frame(
      Metric = "auc",
      Class = NA_character_,
      Reference = auc_data$Reference,
      External = auc_data$External,
      Difference = auc_data$Difference,
      Lower = auc_data$Lower,
      Upper = auc_data$Upper,
      N_valid = NA_integer_,
      N_boot = NA_integer_,
      Method = auc_data$Method,
      stringsAsFactors = FALSE
    )

    summary <- rbind(
      summary,
      auc_row
    )

  }

  rownames(summary) <- NULL

  summary

}

# =============================================================================
# 18. Attach compare confidence intervals
# =============================================================================
#' @keywords internal

.attach_compare_confidence_intervals <- function(
    compare_metrics,
    confidence_intervals,
    average = "macro"
) {

  # Validate inputs
  required_ci_columns <- c(
    "Algorithm", "Metric", "Class",
    "Lower", "Upper"
  )

  missing_columns <- setdiff(
    required_ci_columns,
    colnames(confidence_intervals)
  )

  if (length(missing_columns) > 0) {
    stop(
      "Missing columns in confidence_intervals: ",
      paste(missing_columns, collapse = ", "),
      "."
    )
  }

  # Helper to retrieve a specific confidence interval
  get_interval <- function(
    algorithm,
    metric,
    class = NULL
  ) {

    idx <- which(
      confidence_intervals$Algorithm == algorithm &
        confidence_intervals$Metric == metric
    )

    if (is.null(class)) {

      idx <- idx[
        is.na(confidence_intervals$Class[idx])
      ]

    } else {

      idx <- idx[
        !is.na(confidence_intervals$Class[idx]) &
          as.character(confidence_intervals$Class[idx]) == class
      ]
    }

    if (length(idx) > 1) {
      stop(
        "Multiple confidence intervals found for algorithm '",
        algorithm,
        "', metric '", metric, "'."
      )
    }

    if (length(idx) == 0) {
      return(c(Lower = NA_real_, Upper = NA_real_))
    }

    c(
      Lower = confidence_intervals$Lower[idx],
      Upper = confidence_intervals$Upper[idx]
    )
  }

  # ===============================================
  # 1. Global metrics
  # ===============================================

  summary_metric_map <- c(
    Accuracy = "accuracy",
    Kappa = "kappa",
    Balanced_Accuracy = "macro_recall"
  )

  if (average == "macro") {

    summary_metric_map <- c(
      summary_metric_map,
      Precision = "macro_precision",
      Recall = "macro_recall",
      Specificity = "macro_specificity",
      F1 = "macro_f1"
    )

  } else {

    summary_metric_map <- c(
      summary_metric_map,
      Precision = "weighted_precision",
      Recall = "weighted_recall",
      Specificity = "weighted_specificity",
      F1 = "weighted_f1"
    )
  }

  summary <- compare_metrics$summary

  for (metric_name in names(summary_metric_map)) {

    lower_col <- paste0(metric_name, "_Lower")
    upper_col <- paste0(metric_name, "_Upper")

    summary[[lower_col]] <- NA_real_
    summary[[upper_col]] <- NA_real_

    for (i in seq_len(nrow(summary))) {

      ci <- get_interval(
        algorithm = summary$Algorithm[i],
        metric = summary_metric_map[[metric_name]]
      )

      summary[[lower_col]][i] <- ci["Lower"]
      summary[[upper_col]][i] <- ci["Upper"]
    }
  }

  # ===============================================
  # 2. Classwise metrics
  # ===============================================

  classwise_metric_map <- c(
    Precision = "precision",
    Recall = "recall",
    Specificity = "specificity",
    F1 = "f1"
  )

  classwise <- compare_metrics$classwise

  for (metric_name in names(classwise_metric_map)) {

    lower_col <- paste0(metric_name, "_Lower")
    upper_col <- paste0(metric_name, "_Upper")

    classwise[[lower_col]] <- NA_real_
    classwise[[upper_col]] <- NA_real_

    for (i in seq_len(nrow(classwise))) {

      ci <- get_interval(
        algorithm = classwise$Algorithm[i],
        metric = classwise_metric_map[[metric_name]],
        class = as.character(classwise$Class[i])
      )

      classwise[[lower_col]][i] <- ci["Lower"]
      classwise[[upper_col]][i] <- ci["Upper"]
    }
  }

  # ===============================================
  # 3. Return updated metrics
  # ===============================================

  compare_metrics$summary <- summary
  compare_metrics$classwise <- classwise
  compare_metrics$confidence_intervals <- confidence_intervals

  return(compare_metrics)
}
