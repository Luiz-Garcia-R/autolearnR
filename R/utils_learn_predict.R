# =============================================================================
# LEARN PREDICT HELPERS
# =============================================================================

# =============================================================================
# 1. Prepare model matrix for prediction
# =============================================================================
#' @keywords internal

.prepare_model_matrix <- function(
    new_data,
    predictor_names,
    preprocessing
) {

  # ===============================================
  # 1.1. Validate input
  # ===============================================

  if (!is.data.frame(new_data)) {
    stop("'new_data' must be a data.frame.")
  }

  if (
    !is.character(predictor_names) ||
    length(predictor_names) == 0 ||
    anyNA(predictor_names) ||
    anyDuplicated(predictor_names)
  ) {
    stop("Invalid predictor names stored in frozen model.")
  }

  required_features <- predictor_names

  missing_features <- setdiff(
    required_features,
    colnames(new_data)
  )

  if (length(missing_features) > 0) {
    stop(
      "Missing predictors in 'new_data': ",
      paste(missing_features, collapse = ", ")
    )
  }

  # Keep original predictor order and ignore extra columns
  x <- new_data[
    ,
    required_features,
    drop = FALSE
  ]

  # ===============================================
  # 1.2. Validate preprocessing information
  # ===============================================

  required_fields <- c(
    "encoding_terms",
    "factor_levels",
    "retained_columns"
  )

  missing_fields <- setdiff(
    required_fields,
    names(preprocessing)
  )

  if (length(missing_fields) > 0) {
    stop(
      "Incomplete preprocessing information in frozen model. ",
      "Missing fields: ",
      paste(missing_fields, collapse = ", ")
    )
  }

  factor_levels <- preprocessing$factor_levels

  retained_columns <- preprocessing$retained_columns

  if (
    !is.character(retained_columns) ||
    length(retained_columns) == 0 ||
    anyNA(retained_columns) ||
    anyDuplicated(retained_columns)
  ) {
    stop(
      "Invalid retained columns stored in frozen model."
    )
  }

  # ===============================================
  # 1.3. Validate missing values
  # ===============================================

  missing_values <- names(x)[
    vapply(
      x,
      anyNA,
      logical(1)
    )
  ]

  if (length(missing_values) > 0) {
    stop(
      "'new_data' contains missing values in predictors: ",
      paste(missing_values, collapse = ", ")
    )
  }

  # ===============================================
  # 1.4. Restore factor levels
  # ===============================================

  for (feature in names(factor_levels)) {

    if (!feature %in% required_features) {
      stop(
        "Factor information for predictor '",
        feature,
        "' is inconsistent with frozen predictor names."
      )
    }

    if (
      !is.factor(x[[feature]]) &&
      !is.character(x[[feature]])
    ) {
      stop(
        "Predictor '", feature,
        "' must be a factor or character variable."
      )
    }

    observed <- as.character(x[[feature]])

    unseen <- setdiff(
      unique(observed),
      factor_levels[[feature]]
    )

    if (length(unseen) > 0) {
      stop(
        "New unseen levels detected in predictor '",
        feature,
        "': ",
        paste(unseen, collapse = ", ")
      )
    }

    x[[feature]] <- factor(
      observed,
      levels = factor_levels[[feature]]
    )

  }

  # ===============================================
  # 1.5. Validate numeric predictors
  # ===============================================

  numeric_features <- setdiff(
    required_features,
    names(factor_levels)
  )

  for (feature in numeric_features) {

    if (!is.numeric(x[[feature]])) {
      stop(
        "Predictor '", feature,
        "' must be numeric."
      )
    }

    if (any(!is.finite(x[[feature]]))) {
      stop(
        "Predictor '", feature,
        "' contains non-finite values."
      )
    }

  }

  # ===============================================
  # 1.6. Reconstruct model frame
  # ===============================================

  model_frame <- tryCatch(

    stats::model.frame(
      preprocessing$encoding_terms,
      data = x,
      xlev = factor_levels,
      na.action = stats::na.fail
    ),

    error = function(e) {
      stop(
        "Could not reconstruct model frame: ",
        conditionMessage(e),
        call. = FALSE
      )
    }

  )

  # ===============================================
  # 1.7. Reconstruct model matrix
  # ===============================================

  x_matrix <- tryCatch(

    stats::model.matrix(
      preprocessing$encoding_terms,
      data = model_frame
    ),

    error = function(e) {
      stop(
        "Could not reconstruct model matrix: ",
        conditionMessage(e),
        call. = FALSE
      )
    }

  )

  # ===============================================
  # 1.8. Validate retained columns
  # ===============================================

  missing_columns <- setdiff(
    retained_columns,
    colnames(x_matrix)
  )

  if (length(missing_columns) > 0) {
    stop(
      "Could not reconstruct all retained training columns: ",
      paste(missing_columns, collapse = ", ")
    )
  }

  # Keep exactly the columns and order used during training
  x_matrix <- x_matrix[
    ,
    retained_columns,
    drop = FALSE
  ]

  # ===============================================
  # 1.9. Final validation
  # ===============================================

  if (any(!is.finite(x_matrix))) {
    stop(
      "The reconstructed model matrix contains non-finite values."
    )
  }

  x_matrix

}

# =============================================================================
# 2. Validate prediction data
# =============================================================================
#' @keywords internal

.validate_prediction_data <- function(
    new_data,
    frozen_model
) {

  # ===============================================
  # 2.1. Validate input
  # ===============================================

  if (!is.data.frame(new_data)) {
    stop("'new_data' must be a data.frame.")
  }

  predictor_names <- frozen_model$predictor_names
  predictor_types <- frozen_model$predictor_types

  if (
    !is.character(predictor_names) ||
    length(predictor_names) == 0 ||
    anyNA(predictor_names) ||
    anyDuplicated(predictor_names)
  ) {
    stop("Invalid predictor names stored in frozen model.")
  }

  # ===============================================
  # 2.2. Validate predictor information
  # ===============================================

  if (
    !is.character(predictor_types) ||
    is.null(names(predictor_types)) ||
    !identical(names(predictor_types), predictor_names) ||
    anyNA(predictor_types) ||
    any(!predictor_types %in% c("numeric", "categorical"))
  ) {
    stop("Invalid predictor types stored in frozen model.")
  }

  # ===============================================
  # 2.3. Check required predictors
  # ===============================================

  missing_features <- setdiff(
    predictor_names,
    colnames(new_data)
  )

  if (length(missing_features) > 0) {
    stop(
      "Missing predictors in 'new_data': ",
      paste(missing_features, collapse = ", ")
    )
  }

  # Keep original predictor order and ignore extra columns
  x <- new_data[
    ,
    predictor_names,
    drop = FALSE
  ]

  # ===============================================
  # 2.4. Validate missing values
  # ===============================================

  missing_values <- names(x)[
    vapply(x, anyNA, logical(1))
  ]

  if (length(missing_values) > 0) {
    stop(
      "'new_data' contains missing values in predictors: ",
      paste(missing_values, collapse = ", ")
    )
  }

  # ===============================================
  # 2.5. Validate predictor types
  # ===============================================

  for (feature in predictor_names) {

    expected_type <- predictor_types[[feature]]
    observed <- x[[feature]]

    if (expected_type == "numeric") {

      if (!is.numeric(observed)) {
        stop(
          "Predictor '", feature,
          "' must be numeric or integer."
        )
      }

      if (any(!is.finite(observed))) {
        stop(
          "Predictor '", feature,
          "' contains non-finite values."
        )
      }

    } else if (expected_type == "categorical") {

      if (
        !is.factor(observed) &&
        !is.character(observed)
      ) {
        stop(
          "Predictor '", feature,
          "' must be a factor or character variable."
        )
      }

      # Validate known factor levels
      known_levels <- frozen_model$preprocessing$factor_levels[[feature]]

      if (is.null(known_levels)) {
        stop(
          "Missing factor level information for predictor '",
          feature, "' in frozen model."
        )
      }

      observed_levels <- unique(as.character(observed))

      unseen <- setdiff(
        observed_levels,
        known_levels
      )

      if (length(unseen) > 0) {
        stop(
          "New unseen levels detected in predictor '",
          feature, "': ",
          paste(unseen, collapse = ", ")
        )
      }

      # Restore training factor levels
      x[[feature]] <- factor(
        as.character(observed),
        levels = known_levels
      )

    }

  }

  # ===============================================
  # 2.6. Final validation
  # ===============================================

  x

}

# =============================================================================
# 3. Predict logistic
# =============================================================================
#' @keywords internal

.predict_logistic <- function(
    new_data,
    frozen_model
) {

  # ===============================================
  # 3.1. Reconstruct model matrix
  # ===============================================

  X_new_matrix <- .prepare_model_matrix(
    new_data = new_data,
    predictor_names = frozen_model$predictor_names,
    preprocessing = frozen_model$preprocessing
  )

  # ===============================================
  # 3.2. Generate positive-class probabilities
  # ===============================================

  pred_prob_positive <- stats::predict(
    frozen_model$model,
    newdata = as.data.frame(
      X_new_matrix,
      check.names = FALSE
    ),
    type = "response"
  )

  pred_prob_positive <- as.numeric(
    pred_prob_positive
  )

  # Validate predicted probabilities
  if (
    length(pred_prob_positive) != nrow(X_new_matrix) ||
    any(!is.finite(pred_prob_positive)) ||
    any(pred_prob_positive < 0 | pred_prob_positive > 1)
  ) {

    stop(
      "Logistic regression generated invalid predicted probabilities."
    )

  }

  # ===============================================
  # 3.3. Organize probabilities
  # ===============================================

  target_levels <- frozen_model$response$target_levels

  positive_class <- frozen_model$response$positive_class
  negative_class <- frozen_model$response$negative_class

  probabilities <- matrix(
    NA_real_,
    nrow = nrow(X_new_matrix),
    ncol = length(target_levels),
    dimnames = list(
      rownames(X_new_matrix),
      target_levels
    )
  )

  probabilities[, positive_class] <- pred_prob_positive

  probabilities[, negative_class] <- 1 - pred_prob_positive

  # ===============================================
  # 3.4. Generate predicted classes
  # ===============================================

  threshold <- frozen_model$prediction$threshold

  pred_class <- ifelse(
    probabilities[, positive_class] >= threshold,
    positive_class,
    negative_class
  )

  predictions <- factor(
    pred_class,
    levels = target_levels
  )

  # ===============================================
  # 3.5. Return standardized output
  # ===============================================

  return(
    list(
      predictions = predictions,
      probabilities = probabilities,
      decision_values = NULL
    )
  )

}

# =============================================================================
# 4. PREDICT ELASTIC NET
# =============================================================================
#' @keywords internal

.predict_elastic <- function(
    new_data,
    frozen_model
) {

  # ===============================================
  # 4.1. Reconstruct model matrix
  # ===============================================

  X_new_matrix <- .prepare_model_matrix(
    new_data = new_data,
    predictor_names = frozen_model$predictor_names,
    preprocessing = frozen_model$preprocessing
  )

  # ===============================================
  # 4.2. Generate probabilities
  # ===============================================

  selected_lambda <- frozen_model$algorithm_parameters$selected_lambda

  pred_raw <- stats::predict(
    frozen_model$model,
    newx = X_new_matrix,
    s = selected_lambda,
    type = "response"
  )

  target_levels <- frozen_model$response$target_levels

  model_family <- frozen_model$model_family

  # ===============================================
  # 4.3. Organize probabilities
  # ===============================================

  if (model_family == "binomial") {

    # ---------------------------------------------
    # 4.3.1. Binary classification
    # ---------------------------------------------

    positive_class <- frozen_model$response$positive_class
    negative_class <- frozen_model$response$negative_class

    positive_prob <- as.numeric(pred_raw)

    # Validate predicted probabilities
    if (
      length(positive_prob) != nrow(X_new_matrix) ||
      any(!is.finite(positive_prob)) ||
      any(positive_prob < 0 | positive_prob > 1)
    ) {

      stop(
        "Elastic Net generated invalid predicted probabilities."
      )

    }

    probabilities <- matrix(
      NA_real_,
      nrow = nrow(X_new_matrix),
      ncol = length(target_levels),
      dimnames = list(
        rownames(X_new_matrix),
        target_levels
      )
    )

    probabilities[, positive_class] <- positive_prob

    probabilities[, negative_class] <- 1 - positive_prob

  } else if (model_family == "multinomial") {

    # ---------------------------------------------
    # 4.3.2. Multiclass classification
    # ---------------------------------------------

    # Extract class names from glmnet output
    class_names <- dimnames(pred_raw)[[2]]

    if (
      is.null(class_names) ||
      !setequal(class_names, target_levels)
    ) {

      stop(
        "Multinomial prediction returned unexpected class names."
      )

    }

    # Extract probability matrix
    if (length(dim(pred_raw)) == 3) {

      if (dim(pred_raw)[3] != 1) {

        stop(
          "Unexpected number of lambda values in multinomial prediction."
        )

      }

      pred_matrix <- matrix(
        pred_raw[, , 1],
        nrow = nrow(X_new_matrix),
        ncol = length(class_names)
      )

    } else if (length(dim(pred_raw)) == 2) {

      pred_matrix <- as.matrix(pred_raw)

    } else {

      stop(
        "Unexpected prediction structure returned by multinomial Elastic Net."
      )

    }

    # Align probability columns with original target levels
    probabilities <- pred_matrix[
      ,
      match(target_levels, class_names),
      drop = FALSE
    ]

    colnames(probabilities) <- target_levels
    rownames(probabilities) <- rownames(X_new_matrix)

  } else {

    stop(
      "Unsupported model family in frozen Elastic Net model."
    )

  }

  # ===============================================
  # 4.4. Validate probability matrix
  # ===============================================

  if (
    any(!is.finite(probabilities)) ||
    any(probabilities < 0 | probabilities > 1)
  ) {

    stop(
      "Elastic Net generated invalid probability values."
    )

  }

  probability_sums <- rowSums(probabilities)

  if (
    any(!is.finite(probability_sums)) ||
    any(abs(probability_sums - 1) > 1e-6)
  ) {

    stop(
      "Predicted probabilities do not sum to 1 for all observations."
    )

  }

  # ===============================================
  # 4.5. Generate predicted classes
  # ===============================================

  if (model_family == "binomial") {

    threshold <- frozen_model$prediction$threshold

    pred_class <- ifelse(
      probabilities[, positive_class] >= threshold,
      positive_class,
      negative_class
    )

  } else {

    pred_class <- target_levels[
      max.col(
        probabilities,
        ties.method = "first"
      )
    ]

  }

  predictions <- factor(
    pred_class,
    levels = target_levels
  )

  names(predictions) <- rownames(X_new_matrix)

  # ===============================================
  # 4.6. Return standardized output
  # ===============================================

  return(
    list(
      predictions = predictions,
      probabilities = probabilities,
      decision_values = NULL
    )
  )

}

# =============================================================================
# 5. Predict tree
# =============================================================================
#' @keywords internal

.predict_tree <- function(
    new_data,
    frozen_model
) {

  # ===============================================
  # 5.1. Validate prediction data
  # ===============================================

  X_new <- .validate_prediction_data(
    new_data = new_data,
    frozen_model = frozen_model
  )

  # ===============================================
  # 5.2. Generate predicted probabilities
  # ===============================================

  pred_raw <- stats::predict(
    frozen_model$model,
    newdata = X_new,
    type = "prob"
  )

  probabilities <- as.matrix(pred_raw)

  target_levels <- frozen_model$response$target_levels

  # ===============================================
  # 5.3. Validate probability structure
  # ===============================================

  if (
    is.null(colnames(probabilities)) ||
    !setequal(colnames(probabilities), target_levels)
  ) {
    stop(
      "Decision Tree prediction returned unexpected class names."
    )
  }

  if (
    nrow(probabilities) != nrow(X_new) ||
    ncol(probabilities) != length(target_levels)
  ) {
    stop(
      "Decision Tree prediction returned an unexpected probability matrix dimension."
    )
  }

  # ===============================================
  # 5.4. Align probability columns
  # ===============================================

  probabilities <- probabilities[
    ,
    match(target_levels, colnames(probabilities)),
    drop = FALSE
  ]

  colnames(probabilities) <- target_levels
  rownames(probabilities) <- rownames(X_new)

  # ===============================================
  # 5.5. Validate probability values
  # ===============================================

  if (
    any(!is.finite(probabilities)) ||
    any(probabilities < 0 | probabilities > 1)
  ) {
    stop(
      "Decision Tree generated invalid predicted probabilities."
    )
  }

  probability_sums <- rowSums(probabilities)

  if (
    any(!is.finite(probability_sums)) ||
    any(abs(probability_sums - 1) > 1e-6)
  ) {
    stop(
      "Predicted probabilities do not sum to 1 for all observations."
    )
  }

  # ===============================================
  # 5.6. Generate predicted classes
  # ===============================================

  pred_class <- target_levels[
    max.col(
      probabilities,
      ties.method = "first"
    )
  ]

  predictions <- factor(
    pred_class,
    levels = target_levels
  )

  names(predictions) <- rownames(X_new)

  # ===============================================
  # 5.7. Return standardized output
  # ===============================================

  return(list(
    predictions = predictions,
    probabilities = probabilities,
    decision_values = NULL
  ))

}

# =============================================================================
# 6. Predict rf
# =============================================================================
#' @keywords internal

.predict_rf <- function(
    new_data,
    frozen_model
) {

  # ===============================================
  # 6.1. Validate prediction data
  # ===============================================

  X_new <- .validate_prediction_data(
    new_data = new_data,
    frozen_model = frozen_model
  )

  # ===============================================
  # 6.2. Generate predicted probabilities
  # ===============================================

  pred_raw <- stats::predict(
    frozen_model$model,
    newdata = X_new,
    type = "prob"
  )

  probabilities <- as.matrix(pred_raw)

  target_levels <- frozen_model$response$target_levels

  # ===============================================
  # 6.3. Validate probability structure
  # ===============================================

  if (
    is.null(colnames(probabilities)) ||
    !setequal(colnames(probabilities), target_levels)
  ) {
    stop(
      "Random Forest prediction returned unexpected class names."
    )
  }

  if (
    nrow(probabilities) != nrow(X_new) ||
    ncol(probabilities) != length(target_levels)
  ) {
    stop(
      "Random Forest prediction returned an unexpected probability matrix dimension."
    )
  }

  # ===============================================
  # 6.4. Align probability columns
  # ===============================================

  probabilities <- probabilities[
    ,
    match(target_levels, colnames(probabilities)),
    drop = FALSE
  ]

  colnames(probabilities) <- target_levels
  rownames(probabilities) <- rownames(X_new)

  # ===============================================
  # 6.5. Validate probability values
  # ===============================================

  if (
    any(!is.finite(probabilities)) ||
    any(probabilities < 0 | probabilities > 1)
  ) {
    stop(
      "Random Forest generated invalid predicted probabilities."
    )
  }

  probability_sums <- rowSums(probabilities)

  if (
    any(!is.finite(probability_sums)) ||
    any(abs(probability_sums - 1) > 1e-6)
  ) {
    stop(
      "Predicted probabilities do not sum to 1 for all observations."
    )
  }

  # ===============================================
  # 6.6. Generate predicted classes
  # ===============================================

  pred_class <- target_levels[
    max.col(
      probabilities,
      ties.method = "first"
    )
  ]

  predictions <- factor(
    pred_class,
    levels = target_levels
  )

  names(predictions) <- rownames(X_new)

  # ===============================================
  # 6.7. Return standardized output
  # ===============================================

  return(list(
    predictions = predictions,
    probabilities = probabilities,
    decision_values = NULL
  ))

}

# =============================================================================
# 7. Predict SVM
# =============================================================================
#' @keywords internal

.predict_svm <- function(
    new_data,
    frozen_model
) {

  # ===============================================
  # 7.1. Validate prediction data
  # ===============================================

  X_new <- .validate_prediction_data(
    new_data = new_data,
    frozen_model = frozen_model
  )

  # ===============================================
  # 7.2. Apply frozen preprocessing
  # ===============================================

  preprocessor <- frozen_model$preprocessing$preprocessor

  X_new_processed <- .apply_svm_preprocessor(
    x = X_new,
    preprocessor = preprocessor
  )

  # ===============================================
  # 7.3. Generate native SVM predictions
  # ===============================================

  pred_raw <- stats::predict(
    frozen_model$model,
    newdata = X_new_processed,
    probability = TRUE,
    decision.values = TRUE
  )

  # ===============================================
  # 7.4. Extract predicted classes
  # ===============================================

  target_levels <- frozen_model$response$target_levels

  pred_class <- as.character(pred_raw)

  if (
    length(pred_class) != nrow(X_new_processed) ||
    anyNA(pred_class) ||
    any(!pred_class %in% target_levels)
  ) {
    stop(
      "SVM generated invalid predicted classes."
    )
  }

  predictions <- factor(
    pred_class,
    levels = target_levels
  )

  names(predictions) <- rownames(X_new_processed)

  # ===============================================
  # 7.5. Extract predicted probabilities
  # ===============================================

  pred_prob <- attr(
    pred_raw,
    "probabilities"
  )

  if (is.null(pred_prob)) {
    stop(
      "SVM did not return predicted probabilities. ",
      "Check whether the model was trained with probability = TRUE."
    )
  }

  probabilities <- as.matrix(pred_prob)

  # ===============================================
  # 7.6. Validate probability structure
  # ===============================================

  if (
    is.null(colnames(probabilities)) ||
    !setequal(colnames(probabilities), target_levels)
  ) {
    stop(
      "SVM prediction returned unexpected class names."
    )
  }

  if (
    nrow(probabilities) != nrow(X_new_processed) ||
    ncol(probabilities) != length(target_levels)
  ) {
    stop(
      "SVM prediction returned an unexpected probability matrix dimension."
    )
  }

  # ===============================================
  # 7.7. Align probability columns
  # ===============================================

  probabilities <- probabilities[
    ,
    match(target_levels, colnames(probabilities)),
    drop = FALSE
  ]

  colnames(probabilities) <- target_levels
  rownames(probabilities) <- rownames(X_new_processed)

  # ===============================================
  # 7.8. Validate probability values
  # ===============================================

  if (
    any(!is.finite(probabilities)) ||
    any(probabilities < 0 | probabilities > 1)
  ) {
    stop(
      "SVM generated invalid predicted probabilities."
    )
  }

  probability_sums <- rowSums(probabilities)

  if (
    any(!is.finite(probability_sums)) ||
    any(abs(probability_sums - 1) > 1e-6)
  ) {
    stop(
      "SVM predicted probabilities do not sum to 1 for all observations."
    )
  }

  # ===============================================
  # 7.9. Extract decision values
  # ===============================================

  decision_values <- attr(
    pred_raw,
    "decision.values"
  )

  if (is.null(decision_values)) {
    stop(
      "SVM did not return decision values."
    )
  }

  decision_values <- as.matrix(decision_values)

  if (
    nrow(decision_values) != nrow(X_new_processed) ||
    any(!is.finite(decision_values))
  ) {
    stop(
      "SVM generated invalid decision values."
    )
  }

  rownames(decision_values) <- rownames(X_new_processed)

  # ===============================================
  # 7.10. Return standardized output
  # ===============================================

  return(list(
    predictions = predictions,
    probabilities = probabilities,
    decision_values = decision_values
  ))

}

# =============================================================================
# 8.Validate frozen model
# =============================================================================
#' @keywords internal

.validate_frozen_model <- function(frozen_model) {

  # ===============================================
  # 8.1. Validate general structure
  # ===============================================

  if (!is.list(frozen_model)) {
    stop("'frozen_model' must be a list.")
  }

  supported_algorithms <- c(
    "logistic",
    "elastic_net",
    "tree",
    "random_forest",
    "svm"
  )

  if (
    !is.character(frozen_model$algorithm) ||
    length(frozen_model$algorithm) != 1 ||
    is.na(frozen_model$algorithm) ||
    !frozen_model$algorithm %in% supported_algorithms
  ) {
    stop("Unsupported or invalid frozen model algorithm.")
  }

  algorithm <- frozen_model$algorithm

  expected_classes <- c(
    logistic = "autolearn_logistic",
    elastic_net = "autolearn_elastic_net",
    tree = "autolearn_tree",
    random_forest = "autolearn_random_forest",
    svm = "autolearn_svm"
  )

  if (!inherits(frozen_model, expected_classes[[algorithm]])) {
    stop(
      "Frozen model class is inconsistent with algorithm '",
      algorithm, "'."
    )
  }

  # ===============================================
  # 8.2. Validate required fields
  # ===============================================

  required_fields <- c(
    "schema_version",
    "algorithm",
    "model",
    "model_family",
    "predictor_names",
    "predictor_types",
    "preprocessing",
    "response",
    "prediction",
    "algorithm_parameters",
    "reproducibility"
  )

  missing_fields <- setdiff(
    required_fields,
    names(frozen_model)
  )

  if (length(missing_fields) > 0) {
    stop(
      "Incomplete frozen model. Missing fields: ",
      paste(missing_fields, collapse = ", ")
    )
  }

  # ===============================================
  # 8.3. Validate schema version
  # ===============================================

  if (
    !identical(frozen_model$schema_version, 1L)
  ) {
    stop(
      "Unsupported frozen model schema version."
    )
  }

  # ===============================================
  # 8.4. Validate fitted model
  # ===============================================

  if (is.null(frozen_model$model)) {
    stop("Frozen model does not contain a fitted model.")
  }

  # ===============================================
  # 8.5. Validate model family
  # ===============================================

  if (
    !is.character(frozen_model$model_family) ||
    length(frozen_model$model_family) != 1 ||
    is.na(frozen_model$model_family) ||
    !frozen_model$model_family %in% c(
      "binomial",
      "multinomial"
    )
  ) {
    stop("Invalid model family stored in frozen model.")
  }

  model_family <- frozen_model$model_family

  if (
    model_family == "binomial" &&
    algorithm %in% c("tree", "random_forest", "svm") &&
    length(frozen_model$response$target_levels) != 2
  ) {
    stop("Inconsistent target levels for binomial model.")
  }

  if (
    model_family == "multinomial" &&
    length(frozen_model$response$target_levels) < 3
  ) {
    stop("Inconsistent target levels for multinomial model.")
  }

  # ===============================================
  # 8.6. Validate predictors
  # ===============================================

  predictor_names <- frozen_model$predictor_names
  predictor_types <- frozen_model$predictor_types

  if (
    !is.character(predictor_names) ||
    length(predictor_names) == 0 ||
    anyNA(predictor_names) ||
    anyDuplicated(predictor_names)
  ) {
    stop("Invalid predictor names stored in frozen model.")
  }

  if (
    !is.character(predictor_types) ||
    is.null(names(predictor_types)) ||
    !identical(names(predictor_types), predictor_names) ||
    anyNA(predictor_types) ||
    any(!predictor_types %in% c("numeric", "categorical"))
  ) {
    stop("Invalid predictor types stored in frozen model.")
  }

  # ===============================================
  # 8.7. Validate response information
  # ===============================================

  response <- frozen_model$response

  if (!is.list(response)) {
    stop("Invalid response information in frozen model.")
  }

  if (
    !is.character(response$target_levels) ||
    length(response$target_levels) < 2 ||
    anyNA(response$target_levels) ||
    anyDuplicated(response$target_levels)
  ) {
    stop("Invalid target levels stored in frozen model.")
  }

  if (
    model_family == "binomial" &&
    length(response$target_levels) != 2
  ) {
    stop("Binomial models must contain exactly two target levels.")
  }

  if (
    model_family == "multinomial" &&
    length(response$target_levels) < 3
  ) {
    stop("Multinomial models must contain at least three target levels.")
  }

  if (model_family == "binomial") {

    if (
      !is.character(response$positive_class) ||
      length(response$positive_class) != 1 ||
      is.na(response$positive_class) ||
      !response$positive_class %in% response$target_levels
    ) {
      stop("Invalid positive class stored in frozen model.")
    }

    expected_negative <- setdiff(
      response$target_levels,
      response$positive_class
    )

    if (!identical(response$negative_class, expected_negative)) {
      stop("Invalid negative class stored in frozen model.")
    }

  }

  # ===============================================
  # 8.8. Validate preprocessing
  # ===============================================

  preprocessing <- frozen_model$preprocessing

  if (!is.list(preprocessing)) {
    stop("Invalid preprocessing information in frozen model.")
  }

  expected_methods <- c(
    logistic = "model_matrix",
    elastic_net = "model_matrix",
    tree = "native_rpart",
    random_forest = "native_randomForest",
    svm = "svm_preprocessor"
  )

  if (
    !identical(
      preprocessing$method,
      expected_methods[[algorithm]]
    )
  ) {
    stop(
      "Preprocessing method is inconsistent with algorithm '",
      algorithm, "'."
    )
  }

  # ===============================================
  # 8.9. Validate algorithm-specific preprocessing
  # ===============================================

  if (algorithm %in% c("logistic", "elastic_net")) {

    required_preprocessing <- c(
      "encoding_terms",
      "factor_levels",
      "retained_columns"
    )

    missing_preprocessing <- setdiff(
      required_preprocessing,
      names(preprocessing)
    )

    if (length(missing_preprocessing) > 0) {
      stop(
        "Incomplete model matrix preprocessing information."
      )
    }

    if (
      !inherits(preprocessing$encoding_terms, "terms") ||
      !is.list(preprocessing$factor_levels) ||
      !is.character(preprocessing$retained_columns) ||
      length(preprocessing$retained_columns) == 0 ||
      anyNA(preprocessing$retained_columns) ||
      anyDuplicated(preprocessing$retained_columns)
    ) {
      stop("Invalid model matrix preprocessing information.")
    }

  }

  if (algorithm %in% c("tree", "random_forest")) {

    if (!is.list(preprocessing$factor_levels)) {
      stop("Invalid factor level information.")
    }

  }

  if (algorithm == "svm") {

    preprocessor <- preprocessing$preprocessor

    required_preprocessor <- c(
      "terms",
      "factor_levels",
      "feature_names",
      "predictor_names",
      "dropped_features",
      "centers",
      "scales",
      "scale"
    )

    missing_preprocessor <- setdiff(
      required_preprocessor,
      names(preprocessor)
    )

    if (length(missing_preprocessor) > 0) {
      stop(
        "Incomplete SVM preprocessing information. Missing fields: ",
        paste(missing_preprocessor, collapse = ", ")
      )
    }

    if (
      !inherits(preprocessor$terms, "terms") ||
      !is.list(preprocessor$factor_levels) ||
      !is.character(preprocessor$feature_names) ||
      !identical(preprocessor$predictor_names, predictor_names) ||
      !is.logical(preprocessor$scale) ||
      length(preprocessor$scale) != 1 ||
      anyNA(preprocessor$scale)
    ) {
      stop("Invalid SVM preprocessing information.")
    }

    if (
      !is.numeric(preprocessor$centers) ||
      !is.numeric(preprocessor$scales) ||
      !identical(
        names(preprocessor$centers),
        names(preprocessor$scales)
      ) ||
      any(!is.finite(preprocessor$centers)) ||
      any(!is.finite(preprocessor$scales)) ||
      any(preprocessor$scales <= 0)
    ) {
      stop("Invalid SVM scaling information.")
    }

  }

  # ===============================================
  # 8.10. Validate prediction settings
  # ===============================================

  prediction <- frozen_model$prediction

  if (!is.list(prediction)) {
    stop("Invalid prediction settings in frozen model.")
  }

  if (algorithm %in% c("logistic", "elastic_net")) {

    if (
      !is.numeric(prediction$threshold) ||
      length(prediction$threshold) != 1 ||
      !is.finite(prediction$threshold) ||
      prediction$threshold < 0 ||
      prediction$threshold > 1
    ) {
      stop("Invalid classification threshold in frozen model.")
    }

  }

  # ===============================================
  # 8.11. Return validated model
  # ===============================================

  invisible(frozen_model)

}

# =============================================================================
# 9. Classification performance comparison
# =============================================================================
#' @keywords internal

.compare_classification_performance <- function(
    reference_observed,
    reference_predictions,
    external_observed,
    external_predictions,
    n_boot = 2000,
    conf_level = 0.95,
    seed = NULL
) {

  # ===============================================
  # 9.1. Validate input
  # ===============================================

  validate_evaluation <- function(
    observed,
    predicted,
    label
  ) {

    if (
      !is.factor(observed) ||
      !is.factor(predicted)
    ) {
      stop(
        "'", label,
        "' observed and predicted values must be factors."
      )
    }

    if (
      length(observed) == 0 ||
      length(predicted) != length(observed)
    ) {
      stop(
        "'", label,
        "' observed and predicted values must have the same non-zero length."
      )
    }

    if (
      anyNA(observed) ||
      anyNA(predicted)
    ) {
      stop(
        "'", label,
        "' observed and predicted values cannot contain missing values."
      )
    }

    if (!identical(levels(observed), levels(predicted))) {
      stop(
        "'", label,
        "' observed and predicted values must have identical class levels."
      )
    }

    invisible(TRUE)
  }

  validate_evaluation(
    observed = reference_observed,
    predicted = reference_predictions,
    label = "reference"
  )

  validate_evaluation(
    observed = external_observed,
    predicted = external_predictions,
    label = "external"
  )

  if (
    !identical(
      levels(reference_observed),
      levels(external_observed)
    )
  ) {
    stop(
      "Reference and external evaluations must have identical class levels."
    )
  }

  if (
    length(n_boot) != 1 ||
    is.na(n_boot) ||
    n_boot < 1 ||
    n_boot != as.integer(n_boot)
  ) {
    stop("'n_boot' must be a positive integer.")
  }

  if (
    length(conf_level) != 1 ||
    is.na(conf_level) ||
    conf_level <= 0 ||
    conf_level >= 1
  ) {
    stop("'conf_level' must be between 0 and 1.")
  }

  if (
    !is.null(seed) &&
    (
      length(seed) != 1 ||
      is.na(seed) ||
      !is.numeric(seed)
    )
  ) {
    stop("'seed' must be NULL or a single numeric value.")
  }

  # ===============================================
  # 9.2. Point estimates
  # ===============================================

  reference_results <- .classification_metric_values(
    predicted = reference_predictions,
    observed = reference_observed
  )

  external_results <- .classification_metric_values(
    predicted = external_predictions,
    observed = external_observed
  )

  reference_values <- reference_results$values
  external_values <- external_results$values

  if (
    !identical(
      names(reference_values),
      names(external_values)
    )
  ) {
    stop(
      "Reference and external metric structures are inconsistent."
    )
  }

  # ===============================================
  # 9.3. Preserve random state
  # ===============================================

  if (!is.null(seed)) {

    had_seed <- exists(
      ".Random.seed",
      envir = .GlobalEnv,
      inherits = FALSE
    )

    if (had_seed) {
      old_seed <- get(
        ".Random.seed",
        envir = .GlobalEnv
      )
    }

    on.exit({

      if (had_seed) {

        assign(
          ".Random.seed",
          old_seed,
          envir = .GlobalEnv
        )

      } else if (
        exists(
          ".Random.seed",
          envir = .GlobalEnv,
          inherits = FALSE
        )
      ) {

        rm(
          ".Random.seed",
          envir = .GlobalEnv
        )

      }

    }, add = TRUE)

    set.seed(seed)
  }

  # ===============================================
  # 9.4. Independent bootstrap
  # ===============================================

  n_reference <- length(reference_observed)
  n_external <- length(external_observed)

  metric_names <- names(reference_values)

  bootstrap_differences <- matrix(
    NA_real_,
    nrow = n_boot,
    ncol = length(metric_names),
    dimnames = list(
      NULL,
      metric_names
    )
  )

  for (b in seq_len(n_boot)) {

    # Independent resampling
    reference_index <- sample(
      seq_len(n_reference),
      size = n_reference,
      replace = TRUE
    )

    external_index <- sample(
      seq_len(n_external),
      size = n_external,
      replace = TRUE
    )

    # Reference bootstrap
    reference_boot <- .classification_metric_values(
      predicted = reference_predictions[reference_index],
      observed = reference_observed[reference_index]
    )

    # External bootstrap
    external_boot <- .classification_metric_values(
      predicted = external_predictions[external_index],
      observed = external_observed[external_index]
    )

    # Difference between independent evaluations
    bootstrap_differences[b, ] <-
      external_boot$values - reference_boot$values

  }

  # ===============================================
  # 9.5. Summarize comparison
  # ===============================================

  alpha <- 1 - conf_level

  min_valid <- ceiling(
    0.80 * n_boot
  )

  comparison_list <- lapply(
    seq_along(metric_names),
    function(i) {

      metric_name <- metric_names[i]

      differences <- bootstrap_differences[, i]
      differences <- differences[is.finite(differences)]

      if (length(differences) < min_valid) {

        lower <- NA_real_
        upper <- NA_real_

      } else {

        limits <- stats::quantile(
          differences,
          probs = c(
            alpha / 2,
            1 - alpha / 2
          ),
          names = FALSE,
          na.rm = TRUE
        )

        lower <- limits[1]
        upper <- limits[2]
      }

      # Extract metric and class
      parts <- strsplit(
        metric_name,
        "::",
        fixed = TRUE
      )[[1]]

      metric <- parts[1]

      class <- if (length(parts) == 2) {
        parts[2]
      } else {
        NA_character_
      }

      reference_estimate <- unname(
        reference_values[i]
      )

      external_estimate <- unname(
        external_values[i]
      )

      data.frame(
        Metric = metric,
        Class = class,
        Reference = reference_estimate,
        External = external_estimate,
        Difference = external_estimate - reference_estimate,
        Lower = lower,
        Upper = upper,
        N_valid = length(differences),
        N_boot = n_boot,
        Method = "independent_bootstrap",
        stringsAsFactors = FALSE
      )

    }
  )

  comparison <- do.call(
    rbind,
    comparison_list
  )

  rownames(comparison) <- NULL

  # ===============================================
  # 9.6. Return
  # ===============================================

  list(
    comparison = comparison,
    reference_n = n_reference,
    external_n = n_external,
    conf_level = conf_level,
    n_boot = n_boot
  )

}

# =============================================================================
# 10. Classification AUC comparison
# =============================================================================
#' @keywords internal

.compare_auc <- function(
    reference_roc,
    external_roc,
    conf_level = 0.95
) {

  # ===============================================
  # 10.1. Validate input
  # ===============================================

  if (
    !inherits(reference_roc, "roc") ||
    !inherits(external_roc, "roc")
  ) {
    stop(
      "'reference_roc' and 'external_roc' must be pROC ROC objects."
    )
  }

  if (
    length(conf_level) != 1 ||
    is.na(conf_level) ||
    conf_level <= 0 ||
    conf_level >= 1
  ) {
    stop("'conf_level' must be between 0 and 1.")
  }

  # Verify class alignment
  if (
    !identical(
      reference_roc$levels,
      external_roc$levels
    )
  ) {
    stop(
      "Reference and external ROC objects must have identical class levels."
    )
  }

  # Verify ROC direction
  if (
    !identical(
      reference_roc$direction,
      external_roc$direction
    )
  ) {
    stop(
      "Reference and external ROC objects must have identical directions."
    )
  }

  # ===============================================
  # 10.2. Extract AUC estimates
  # ===============================================

  reference_auc <- as.numeric(
    pROC::auc(reference_roc)
  )

  external_auc <- as.numeric(
    pROC::auc(external_roc)
  )

  difference <- external_auc - reference_auc

  # ===============================================
  # 10.3. Estimate independent DeLong variances
  # ===============================================

  reference_variance <- as.numeric(
    pROC::var(
      reference_roc,
      method = "delong"
    )
  )

  external_variance <- as.numeric(
    pROC::var(
      external_roc,
      method = "delong"
    )
  )

  if (
    length(reference_variance) != 1 ||
    length(external_variance) != 1 ||
    !is.finite(reference_variance) ||
    !is.finite(external_variance) ||
    reference_variance < 0 ||
    external_variance < 0
  ) {
    stop(
      "Unable to estimate valid DeLong variances for both ROC curves."
    )
  }

  # Independent variance of the difference
  difference_se <- sqrt(
    reference_variance + external_variance
  )

  if (
    !is.finite(difference_se) ||
    difference_se <= 0
  ) {
    stop(
      "The standard error of the AUC difference could not be estimated."
    )
  }

  # ===============================================
  # 10.4. Confidence interval for AUC difference
  # ===============================================

  z_critical <- stats::qnorm(
    1 - (1 - conf_level) / 2
  )

  lower <- difference - z_critical * difference_se
  upper <- difference + z_critical * difference_se

  # ===============================================
  # 10.5. Independent DeLong test
  # ===============================================

  test <- pROC::roc.test(
    reference_roc,
    external_roc,
    paired = FALSE,
    method = "delong",
    alternative = "two.sided"
  )

  # ===============================================
  # 10.6. Return
  # ===============================================

  comparison <- data.frame(
    Metric = "auc",
    Reference = reference_auc,
    External = external_auc,
    Difference = difference,
    SE = difference_se,
    Lower = lower,
    Upper = upper,
    Statistic = unname(test$statistic),
    P_value = test$p.value,
    Method = "independent_delong",
    stringsAsFactors = FALSE
  )

  list(
    comparison = comparison,
    test = test,
    reference_variance = reference_variance,
    external_variance = external_variance,
    conf_level = conf_level
  )

}
