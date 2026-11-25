# =============================================================================
# LEARN DESING HELPERS
# =============================================================================


# =============================================================================
# 1. Build predictor matrix
# =============================================================================
#' @keywords internal

.build_predictor_matrix <- function(
    data,
    target_var,
    sample_col = NULL
) {

  # ===============================================
  # 1.1. Validate data
  # ===============================================

  if (!is.data.frame(data)) {
    stop("'data' must be a data.frame.")
  }

  if (nrow(data) == 0 || ncol(data) == 0) {
    stop("'data' must contain at least one sample and one variable.")
  }

  # Work on a copy
  data <- as.data.frame(data)

  # ===============================================
  # 1.2. Sample identifiers
  # ===============================================

  if (!is.null(sample_col)) {

    if (
      !is.character(sample_col) ||
      length(sample_col) != 1 ||
      is.na(sample_col) ||
      !sample_col %in% colnames(data)
    ) {
      stop("'sample_col' must identify a column in data.")
    }

    sample_ids <- as.character(data[[sample_col]])

    if (anyNA(sample_ids) || any(trimws(sample_ids) == "")) {
      stop("Sample identifiers cannot contain missing or empty values.")
    }

    if (anyDuplicated(sample_ids)) {
      stop("Duplicated sample identifiers detected.")
    }

    rownames(data) <- sample_ids

    data[[sample_col]] <- NULL

  } else {

    # Preserve existing row names when available
    sample_ids <- rownames(data)

    if (
      is.null(sample_ids) ||
      identical(sample_ids, as.character(seq_len(nrow(data))))
    ) {

      sample_ids <- paste0(
        "Sample_",
        seq_len(nrow(data))
      )
    }

    if (anyDuplicated(sample_ids)) {
      stop("Duplicated sample identifiers detected.")
    }

    rownames(data) <- sample_ids
  }

  # ===============================================
  # 1.3. Target variable
  # ===============================================

  if (
    !is.character(target_var) ||
    length(target_var) != 1 ||
    is.na(target_var) ||
    !nzchar(trimws(target_var)) ||
    !target_var %in% colnames(data)
  ) {
    stop("'target_var' must identify a column in data.")
  }

  y <- data[[target_var]]

  data[[target_var]] <- NULL

  # ===============================================
  # 1.4. Predictor matrix
  # ===============================================

  predictor_matrix <- data

  if (ncol(predictor_matrix) == 0) {
    stop(
      "No predictors available after removing sample and target columns."
    )
  }

  list(
    predictor_matrix = predictor_matrix,
    response = y,
    predictor_sources = "data"
  )
}


# =============================================================================
# 3. Compute diagnostics
# =============================================================================
#' @keywords internal

.compute_diagnostics <- function(
    predictor_matrix,
    y
) {

  n_samples <- nrow(predictor_matrix)
  n_predictors <- ncol(predictor_matrix)

  # ===============================================
  # 3.1. Basic structure
  # ===============================================

  na_count <- vapply(
    predictor_matrix,
    function(x) sum(is.na(x)),
    numeric(1)
  )

  total_missing <- sum(na_count)

  predictors_with_na <- sum(na_count > 0)

  total_values <- n_samples * n_predictors

  overall_missing_fraction <- if (total_values > 0) {
    total_missing / total_values
  } else {
    NA_real_
  }

  # ===============================================
  # 3.2. Zero variance
  # ===============================================

  zero_variance <- vapply(
    predictor_matrix,
    function(x) {

      x <- x[!is.na(x)]

      if (length(x) <= 1) {
        return(TRUE)
      }

      if (is.numeric(x)) {
        return(
          isTRUE(
            is.finite(stats::var(x)) &&
              stats::var(x) < 1e-8
          )
        )
      }

      length(unique(x)) <= 1
    },
    logical(1)
  )

  n_zero_variance <- sum(zero_variance)

  # ===============================================
  # 3.3. Dataset structure
  # ===============================================

  duplicated_samples <- if (!is.null(rownames(predictor_matrix))) {
    sum(duplicated(rownames(predictor_matrix)))
  } else {
    NA_integer_
  }

  high_dimensional <- n_predictors > n_samples

  numeric_predictors <- sum(
    vapply(predictor_matrix, is.numeric, logical(1))
  )

  categorical_predictors <- n_predictors - numeric_predictors

  mixed_predictors <- (
    numeric_predictors > 0 &&
      categorical_predictors > 0
  )

  low_information_ratio <- mean(
    vapply(
      predictor_matrix,
      function(x) {

        if (is.numeric(x)) {
          return(
            length(unique(x[!is.na(x)])) <= 2
          )
        }

        FALSE
      },
      logical(1)
    )
  )

  # ===============================================
  # 3.4. Correlation redundancy
  # ===============================================

  correlation_threshold <- 0.95

  numeric_predictor_data <- predictor_matrix[
    ,
    vapply(predictor_matrix, is.numeric, logical(1)),
    drop = FALSE
  ]

  # Remove zero-variance predictors
  if (ncol(numeric_predictor_data) > 0) {

    valid_numeric_predictors <- vapply(
      numeric_predictor_data,
      function(x) {

        x <- x[!is.na(x)]

        length(x) > 1 &&
          is.finite(stats::var(x)) &&
          stats::var(x) >= 1e-8
      },
      logical(1)
    )

    numeric_predictor_data <- numeric_predictor_data[
      ,
      valid_numeric_predictors,
      drop = FALSE
    ]
  }

  # Calculate correlations
  if (ncol(numeric_predictor_data) >= 2) {

    correlation_matrix <- stats::cor(
      numeric_predictor_data,
      use = "pairwise.complete.obs"
    )

    correlation_indices <- which(
      upper.tri(correlation_matrix) &
        is.finite(correlation_matrix) &
        abs(correlation_matrix) > correlation_threshold,
      arr.ind = TRUE
    )

    high_correlation_details <- data.frame(
      Predictor_1 = rownames(correlation_matrix)[
        correlation_indices[, 1]
      ],
      Predictor_2 = colnames(correlation_matrix)[
        correlation_indices[, 2]
      ],
      Correlation = correlation_matrix[correlation_indices],
      stringsAsFactors = FALSE
    )

    if (nrow(high_correlation_details) > 0) {

      high_correlation_details <- high_correlation_details[
        order(
          abs(high_correlation_details$Correlation),
          decreasing = TRUE
        ),
        ,
        drop = FALSE
      ]

      rownames(high_correlation_details) <- NULL
    }

  } else {

    high_correlation_details <- data.frame(
      Predictor_1 = character(0),
      Predictor_2 = character(0),
      Correlation = numeric(0),
      stringsAsFactors = FALSE
    )
  }

  high_correlation_pairs <- nrow(high_correlation_details)

  # ===============================================
  # 3.5. Missingness level
  # ===============================================

  missingness_level <- if (is.na(overall_missing_fraction)) {

    NA_character_

  } else if (overall_missing_fraction == 0) {

    "none"

  } else if (overall_missing_fraction <= 0.01) {

    "low"

  } else if (overall_missing_fraction <= 0.05) {

    "moderate"

  } else if (overall_missing_fraction <= 0.20) {

    "high"

  } else {

    "extreme"

  }

  # ===============================================
  # 3.6. Classification diagnostics
  # ===============================================

  class_distribution <- table(
    as.character(y),
    useNA = "no"
  )

  n_classes <- length(class_distribution)

  balance_ratio <- min(class_distribution) /
    max(class_distribution)

  balance <- if (balance_ratio >= 0.5) {
    "Balanced"
  } else {
    "Unbalanced"
  }

  # ===============================================
  # 3.7. Warnings
  # ===============================================

  warnings <- character(0)

  if (total_missing > 0) {
    warnings <- c(
      warnings,
      "Missing values detected in predictors"
    )
  }

  if (n_zero_variance > 0) {
    warnings <- c(
      warnings,
      "Zero variance predictors detected"
    )
  }

  if (anyNA(y)) {
    warnings <- c(
      warnings,
      "Missing values detected in the target variable"
    )
  }

  # ===============================================
  # 3.8. Return
  # ===============================================

  list(
    n_samples = n_samples,
    n_predictors = n_predictors,
    numeric_predictors = numeric_predictors,
    categorical_predictors = categorical_predictors,
    overall_missing_fraction = overall_missing_fraction,
    predictors_with_na = predictors_with_na,
    zero_variance = zero_variance,
    n_zero_variance = n_zero_variance,
    duplicated_samples = duplicated_samples,
    high_dimensional = high_dimensional,
    mixed_predictors = mixed_predictors,
    low_information_ratio = low_information_ratio,
    high_correlation_pairs = high_correlation_pairs,
    high_correlation_details = high_correlation_details,
    correlation_threshold = correlation_threshold,
    missingness_level = missingness_level,
    class_distribution = class_distribution,
    balance = balance,
    n_classes = n_classes,
    warnings = warnings
  )
}

# =============================================================================
# 4. Recommend model
# =============================================================================
#' @keywords internal

.recommend_model <- function(diagnostics) {

  recommended_function <- NULL
  alternatives <- character(0)
  reason <- NULL

  high_dim <- diagnostics$high_dimensional
  mixed <- diagnostics$mixed_predictors
  balance <- diagnostics$balance

  if (high_dim && !mixed) {

    recommended_function <- "learn_elastic_net"
    alternatives <- c("learn_svm", "learn_logistic")

    reason <- paste(
      "High-dimensional classification.",
      "Elastic Net provides regularization and feature selection."
    )

  } else if (mixed) {

    recommended_function <- "learn_tree"
    alternatives <- c("learn_random_forest", "learn_logistic")

    reason <- paste(
      "Mixed numeric and categorical predictors.",
      "Tree-based models can accommodate both predictor types."
    )

  } else if (identical(balance, "Unbalanced")) {

    recommended_function <- "learn_random_forest"
    alternatives <- c("learn_tree", "learn_logistic", "learn_svm")

    reason <- paste(
      "Class imbalance detected.",
      "Evaluate class-specific performance and avoid relying",
      "on accuracy alone."
    )

  } else {

    recommended_function <- "learn_logistic"
    alternatives <- c(
      "learn_elastic_net",
      "learn_tree",
      "learn_random_forest",
      "learn_svm"
    )

    reason <- "General-purpose classification with a linear baseline."
  }

  list(
    available = TRUE,
    recommended_function = recommended_function,
    alternatives = alternatives,
    reason = reason
  )
}
