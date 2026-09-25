# ============================
# Check dependencies
# ============================
#' @keywords internal

.check_dependencies <- function(pkgs, bioc = FALSE) {

  missing <- pkgs[
    !vapply(pkgs, requireNamespace,
            logical(1), quietly = TRUE)
  ]

  if (length(missing) > 0) {

    installer <- if (bioc) {
      paste0(
        "BiocManager::install(c(",
        paste(sprintf('"%s"', missing), collapse = ", "),
        "))"
      )
    } else {
      paste0(
        "install.packages(c(",
        paste(sprintf('"%s"', missing), collapse = ", "),
        "))"
      )
    }

    stop(
      paste0(
        "Missing required packages: ",
        paste(missing, collapse = ", "),
        "\n\nInstall using:\n",
        installer
      ),
      call. = FALSE
    )
  }
}

# ============================
# build_predictor_matrix()
# ============================
#' @keywords internal

build_predictor_matrix <- function(
    feature_data = NULL,
    metadata = NULL,
    predictors = c("feature_data", "metadata"),
    output = NULL,
    sample_column = "Sample"
){

  predictors <- match.arg(
    predictors,
    choices = c("feature_data", "metadata"),
    several.ok = TRUE
  )

  predictor_matrix <- NULL
  predictor_sources <- character()

  # -----------------------------
  # feature_data
  # -----------------------------

  if ("feature_data" %in% predictors) {

    if (is.null(feature_data)) {
      stop("'feature_data' requested but is NULL.")
    }

    if (sample_column %in% colnames(feature_data)) {
      rownames(feature_data) <- feature_data[[sample_column]]
      feature_data[[sample_column]] <- NULL
    }

    predictor_matrix <- feature_data
    predictor_sources <- c(predictor_sources, "feature_data")
  }

  # -----------------------------
  # metadata
  # -----------------------------

  if ("metadata" %in% predictors) {

    if (is.null(metadata)) {
      stop("'metadata' requested but is NULL.")
    }

    if (!(sample_column %in% colnames(metadata))) {
      stop("sample_column not found in metadata.")
    }

    rownames(metadata) <- metadata[[sample_column]]
    meta <- metadata
    meta[[sample_column]] <- NULL

    if (!is.null(output)) {
      meta[[output]] <- NULL
    }

    if (is.null(predictor_matrix)) {
      predictor_matrix <- meta
    } else {
      predictor_matrix <- cbind(predictor_matrix, meta)
    }

    predictor_sources <- c(predictor_sources, "metadata")
  }

  # -----------------------------
  # return
  # -----------------------------

  list(
    predictor_matrix = predictor_matrix,
    predictor_sources = predictor_sources
  )
}


# ============================
# detect_task()
# ============================
#' @keywords internal

detect_task <- function(metadata, output = NULL, objective = NULL){

  if (!is.null(objective)) {
    return(match.arg(objective, c("classification", "regression", "clustering")))
  }

  if (is.null(output)) {
    return("clustering")
  }

  if (is.null(metadata)) {
    stop("metadata required when output is provided.")
  }

  y <- metadata[[output]]

  if (is.factor(y) || is.character(y)) {
    return("classification")
  }

  if (is.numeric(y)) {
    return("regression")
  }

  stop("Cannot infer task from output.")
}


# ============================
# compute_diagnostics()
# ============================
#' @keywords internal

compute_diagnostics <- function(predictor_matrix, metadata = NULL, output = NULL){

  n_samples <- nrow(predictor_matrix)
  n_predictors <- ncol(predictor_matrix)

  # -----------------------------
  # basic structure
  # -----------------------------

  na_count <- sapply(predictor_matrix, function(x) sum(is.na(x)))
  total_missing <- sum(na_count)

  predictors_with_na <- sum(na_count > 0)

  overall_missing_fraction <- total_missing / (n_samples * n_predictors)

  zero_variance <- sapply(predictor_matrix, function(x){
    if (!is.numeric(x)) return(FALSE)
    var(x, na.rm = TRUE) < 1e-8
  })

  n_zero_variance <- sum(zero_variance)

  duplicated_samples <- sum(duplicated(rownames(predictor_matrix)))

  high_dimensional <- n_predictors > n_samples

  numeric_predictors <- sum(sapply(predictor_matrix, is.numeric))
  categorical_predictors <- n_predictors - numeric_predictors

  mixed_predictors <- numeric_predictors > 0 & categorical_predictors > 0

  low_information_ratio <- mean(sapply(predictor_matrix, function(x){
    if (is.numeric(x)) {
      length(unique(x[!is.na(x)])) <= 2
    } else {
      FALSE
    }
  }))

  # -----------------------------
  # correlation redundancy
  # -----------------------------

  high_correlation_pairs <- 0

  num_mat <- predictor_matrix[, sapply(predictor_matrix, is.numeric), drop = FALSE]

  if (ncol(num_mat) > 1) {

    cor_matrix <- stats::cor(num_mat, use = "pairwise.complete.obs")

    high_correlation_pairs <- sum(
      abs(cor_matrix[upper.tri(cor_matrix)]) > 0.95
    )
  }

  # -----------------------------
  # sparsity
  # -----------------------------

  sparsity_level <- cut(
    overall_missing_fraction,
    breaks = c(0, 0.01, 0.05, 0.2, 1),
    labels = c("low", "moderate", "high", "extreme")
  )

  # -----------------------------
  # classification diagnostics
  # -----------------------------

  class_distribution <- NULL
  balance <- NA
  n_classes <- NULL

  if (!is.null(metadata) && !is.null(output)) {

    if (output %in% colnames(metadata)) {

      y <- metadata[[output]]

      if (is.factor(y) || is.character(y)) {

        class_distribution <- table(y)
        n_classes <- length(class_distribution)

        balance_ratio <- min(class_distribution) / max(class_distribution)

        balance <- ifelse(balance_ratio >= 0.5, "Balanced", "Unbalanced")
      }
    }
  }

  # -----------------------------
  # regression diagnostics
  # -----------------------------

  regression_summary <- NULL

  if (!is.null(metadata) && !is.null(output)) {

    y <- metadata[[output]]

    if (is.numeric(y)) {

      regression_summary <- list(
        min = min(y, na.rm = TRUE),
        max = max(y, na.rm = TRUE),
        mean = mean(y, na.rm = TRUE),
        median = stats::median(y, na.rm = TRUE),
        sd = stats::sd(y, na.rm = TRUE)
      )
    }
  }

  # -----------------------------
  # Warnings
  # -----------------------------

  warnings <- character(0)

  if (total_missing > 0) {
    warnings <- c(warnings, "Missing values detected")
  }

  if (n_zero_variance > 0) {
    warnings <- c(warnings, "Zero variance predictors detected")
  }

  # -----------------------------
  # return
  # -----------------------------

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
    sparsity_level = sparsity_level,
    class_distribution = class_distribution,
    balance = balance,
    n_classes = n_classes,
    regression_summary = regression_summary,
    warnings = warnings
  )
}


# ============================
# recommend_model()
# ============================
#' @keywords internal

recommend_model <- function(diagnostics, task){

  high_dim <- diagnostics$high_dimensional
  mixed <- diagnostics$mixed_predictors
  balance <- diagnostics$balance
  sparsity <- diagnostics$sparsity_level

  recommended_function <- NULL
  alternatives <- NULL
  reason <- NULL

  if (task == "classification") {

    if (high_dim && sparsity %in% c("high", "extreme")) {

      recommended_function <- "learn.elasticnet"
      alternatives <- c("learn.forest", "learn.xgboost", "learn.svm")
      reason <- "High-dimensional sparse classification"

    } else if (high_dim) {

      recommended_function <- "learn.elasticnet"
      alternatives <- c("learn.forest", "learn.svm")
      reason <- "High-dimensional data"

    } else if (balance == "Unbalanced") {

      recommended_function <- "learn.forest"
      alternatives <- c("learn.xgboost", "learn.svm")
      reason <- "Class imbalance"

    } else if (mixed) {

      recommended_function <- "learn.xgboost"
      alternatives <- c("learn.forest", "learn.svm")
      reason <- "Mixed predictors"

    } else {

      recommended_function <- "learn.logistic"
      alternatives <- c("learn.forest", "learn.svm")
      reason <- "Simple classification"
    }
  }

  if (task == "regression") {

    if (high_dim) {

      recommended_function <- "learn.elasticnet"
      alternatives <- c("learn.forest", "learn.xgboost")
      reason <- "High-dimensional regression"

    } else if (mixed) {

      recommended_function <- "learn.forest"
      alternatives <- c("learn.xgboost", "learn.elasticnet")
      reason <- "Mixed predictors"

    } else {

      recommended_function <- "learn.linear"
      alternatives <- c("learn.elasticnet", "learn.forest")
      reason <- "Simple regression"
    }
  }

  if (task == "clustering") {

    if (mixed) {

      recommended_function <- "learn.gower_hclust"
      alternatives <- c("learn.pam", "learn.dbscan")
      reason <- "Mixed data types"

    } else if (high_dim) {

      recommended_function <- "learn.hclust"
      alternatives <- c("learn.spectral", "learn.gmm")
      reason <- "High-dimensional data"

    } else {

      recommended_function <- "learn.kmeans"
      alternatives <- c("learn.hclust", "learn.gmm")
      reason <- "Numeric data"
    }
  }

  list(
    recommended_function = recommended_function,
    alternatives = alternatives,
    reason = reason
  )
}

