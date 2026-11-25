# =============================================================================
# GENERIC HELPERS
# =============================================================================

# =============================================================================
# 1. Check dependencies
# =============================================================================
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

# =============================================================================
# 2. Safe division
# =============================================================================
#' @keywords internal

.safe_div <- function(num, den) {

  if (length(num) != length(den)) {
    stop("num and den must have the same length.")
  }

  out <- as.numeric(num / den)

  out[
    is.na(num) |
      is.na(den) |
      den == 0
  ] <- NA_real_

  out
}

# =============================================================================
# 3. Safe mean
# =============================================================================
#' @keywords internal

.safe_mean <- function(x) {

  if (length(x) == 0 || anyNA(x)) {
    return(NA_real_)
  }

  mean(x)
}

# =============================================================================
# 4. Make stratified split
# =============================================================================
#' @keywords internal

.make_stratified_split <- function(
    y,
    p = 0.7
) {

  # Validate proportion
  if (
    length(p) != 1 ||
    !is.numeric(p) ||
    !is.finite(p) ||
    p <= 0 ||
    p >= 1
  ) {
    stop("p must be a single numeric value between 0 and 1.")
  }

  # Validate target
  if (length(y) == 0) {
    stop("y cannot be empty.")
  }

  if (anyNA(y)) {
    stop("y cannot contain missing values.")
  }

  if (length(unique(y)) < 2) {
    stop("y must contain at least two classes.")
  }

  groups <- split(
    seq_along(y),
    droplevels(factor(y))
  )

  # Validate class sizes
  class_sizes <- lengths(groups)

  if (any(class_sizes < 2)) {
    stop(
      "Each class must contain at least 2 samples."
    )
  }

  # Stratified sampling
  idx_train <- unlist(
    lapply(groups, function(idx) {

      n_class <- length(idx)

      n_train <- floor(n_class * p)

      n_train <- max(
        1,
        min(n_train, n_class - 1)
      )

      sample(idx, n_train)

    }),
    use.names = FALSE
  )

  idx_train <- sort(idx_train)

  idx_test <- setdiff(
    seq_along(y),
    idx_train
  )

  list(
    train = idx_train,
    test = idx_test
  )
}

# =============================================================================
# 5. Prepare classification data
# =============================================================================
#' @keywords internal

.prepare_classification_data <- function(
    data,
    target_var,
    feature_subset = NULL,
    sample_col = "Sample",
    min_samples = 10,
    min_class_n = 2
) {

  # =============================================
  # 5.1. Validate input
  # =============================================

  if (!is.data.frame(data)) {
    stop("data must be a data.frame.")
  }

  data <- as.data.frame(
    data,
    stringsAsFactors = FALSE
  )

  # =============================================
  # 5.2. Validate target var
  # =============================================

  if (
    length(target_var) != 1 ||
    is.na(target_var) ||
    !is.character(target_var) ||
    !nzchar(target_var) ||
    !target_var %in% colnames(data)
  ) {
    stop("target_var must identify one existing column.")
  }

  # =============================================
  # 5.3. Validate feature subset
  # =============================================

  if (!is.null(feature_subset)) {

    if (
      !is.character(feature_subset) ||
      anyNA(feature_subset) ||
      any(!nzchar(feature_subset))
    ) {
      stop("feature_subset must contain valid column names.")
    }

    if (length(feature_subset) == 0) {
      stop("feature_subset cannot be empty.")
    }

    if (anyDuplicated(feature_subset)) {
      stop("feature_subset contains duplicated variables.")
    }
  }

  # =============================================
  # 5.4. Validate sample identifier
  # =============================================

  if (!is.null(sample_col)) {

    if (
      length(sample_col) != 1 ||
      is.na(sample_col) ||
      !is.character(sample_col) ||
      !nzchar(sample_col)
    ) {
      stop("sample_col must be NULL or a single column name.")
    }

    if (!sample_col %in% colnames(data)) {

      warning(
        "sample_col was not found. ",
        "No sample identifier will be used."
      )

      sample_col <- NULL
    }
  }

  # =============================================
  # 5.5. Validate missing target values
  # =============================================

  if (anyNA(data[[target_var]])) {
    stop(
      "'data' contains missing values in target variable: ",
      target_var
    )
  }

  # =============================================
  # 5.6. Validate sample thresholds
  # =============================================

  if (
    length(min_samples) != 1 ||
    !is.numeric(min_samples) ||
    !is.finite(min_samples) ||
    min_samples < 1 ||
    min_samples != as.integer(min_samples)
  ) {
    stop("min_samples must be a positive integer.")
  }

  if (
    length(min_class_n) != 1 ||
    !is.numeric(min_class_n) ||
    !is.finite(min_class_n) ||
    min_class_n < 1 ||
    min_class_n != as.integer(min_class_n)
  ) {
    stop("min_class_n must be a positive integer.")
  }

  # =============================================
  # 5.7. Define target
  # =============================================

  y <- droplevels(
    factor(data[[target_var]])
  )

  if (nlevels(y) < 2) {
    stop("Classification requires at least two classes.")
  }

  # =============================================
  # 5.8. Sample identifiers
  # =============================================

  if (!is.null(sample_col)) {

    sample_ids <- as.character(
      data[[sample_col]]
    )

    if (anyNA(sample_ids) || anyDuplicated(sample_ids)) {
      stop(
        "Sample identifiers must be unique and non-missing."
      )
    }

  } else {

    sample_ids <- as.character(
      seq_len(nrow(data))
    )
  }

  # =============================================
  # 5.9. Predictor selection
  # =============================================

  excluded_cols <- c(
    sample_col,
    target_var
  )

  available_features <- setdiff(
    colnames(data),
    excluded_cols
  )

  if (length(available_features) == 0) {
    stop("No predictor variables available.")
  }

  if (!is.null(feature_subset)) {

    if (anyDuplicated(feature_subset)) {
      stop("feature_subset contains duplicated variables.")
    }

    missing_features <- setdiff(
      feature_subset,
      available_features
    )

    if (length(missing_features) > 0) {
      stop(
        "Unknown features: ",
        paste(missing_features, collapse = ", ")
      )
    }

    predictor_cols <- feature_subset

  } else {

    predictor_cols <- available_features
  }

  X <- data[
    ,
    predictor_cols,
    drop = FALSE
  ]

  if (ncol(X) == 0) {
    stop("No predictor variables selected.")
  }

  # =============================================
  # 5.10. Validate missing predictor values
  # =============================================

  missing_values <- names(X)[
    vapply(
      X,
      anyNA,
      logical(1)
    )
  ]

  if (length(missing_values) > 0) {
    stop(
      "'data' contains missing values in predictors: ",
      paste(missing_values, collapse = ", ")
    )
  }

  # =============================================
  # 5.11. Validate sample thresholds
  # =============================================

  if (nrow(X) < min_samples) {
    stop("Too few samples.")
  }

  if (nlevels(y) < 2) {
    stop("Classification requires at least two classes.")
  }

  if (any(table(y) < min_class_n)) {
    stop(
      "Some classes contain fewer than ",
      min_class_n,
      " samples."
    )
  }

  # =============================================
  # 5.12. Assign sample identifiers
  # =============================================

  rownames(X) <- sample_ids

  # =============================================
  # 5.13. Output
  # =============================================

  list(
    x = X,
    y = y,
    sample_ids = sample_ids,
    features = colnames(X),
    classes = levels(y),
    n_samples = nrow(X)
  )
}

# =============================================================================
# 6. Classification metric values
# =============================================================================
#' @keywords internal

.classification_metric_values <- function(
    predicted,
    observed
) {

  cm_tab <- table(
    Predicted = predicted,
    Observed = observed
  )

  total <- sum(cm_tab)

  # Accuracy
  accuracy <- sum(diag(cm_tab)) / total

  # Cohen's Kappa
  observed_agreement <- accuracy

  expected_agreement <- sum(
    rowSums(cm_tab) * colSums(cm_tab)
  ) / total^2

  kappa <- .safe_div(
    observed_agreement - expected_agreement,
    1 - expected_agreement
  )

  # Per-class metrics
  class_metrics <- lapply(
    levels(observed),
    function(cl) {

      tp <- cm_tab[cl, cl]

      fp <- sum(cm_tab[cl, ]) - tp
      fn <- sum(cm_tab[, cl]) - tp
      tn <- total - tp - fp - fn

      precision <- .safe_div(tp, tp + fp)
      recall <- .safe_div(tp, tp + fn)
      specificity <- .safe_div(tn, tn + fp)

      f1 <- .safe_div(
        2 * tp,
        2 * tp + fp + fn
      )

      data.frame(
        Class = cl,
        Precision = precision,
        Recall = recall,
        Specificity = specificity,
        F1 = f1,
        stringsAsFactors = FALSE
      )
    }
  )

  class_metrics <- do.call(
    rbind,
    class_metrics
  )

  rownames(class_metrics) <- NULL

  # Macro metrics
  macro_precision <- .safe_mean(
    class_metrics$Precision
  )

  macro_recall <- .safe_mean(
    class_metrics$Recall
  )

  macro_f1 <- .safe_mean(
    class_metrics$F1
  )

  macro_specificity <- .safe_mean(
    class_metrics$Specificity
  )

  # Combine metrics
  values <- c(
    accuracy = accuracy,
    kappa = kappa,
    macro_precision = macro_precision,
    macro_recall = macro_recall,
    macro_specificity = macro_specificity,
    macro_f1 = macro_f1
  )

  # Add classwise metrics
  for (i in seq_len(nrow(class_metrics))) {

    cl <- class_metrics$Class[i]

    values[paste0("precision::", cl)] <-
      class_metrics$Precision[i]

    values[paste0("recall::", cl)] <-
      class_metrics$Recall[i]

    values[paste0("specificity::", cl)] <-
      class_metrics$Specificity[i]

    values[paste0("f1::", cl)] <-
      class_metrics$F1[i]
  }

  list(
    values = values,
    class_metrics = class_metrics
  )
}

# =============================================================================
# 7. Classification metrics
# =============================================================================
#' @keywords internal

.classification_metrics <- function(
    predicted,
    observed,
    n_boot = 2000,
    conf_level = 0.95,
    seed = NULL
) {

  # Validate input
  if (length(predicted) != length(observed)) {
    stop("predicted and observed must have the same length.")
  }

  if (length(predicted) == 0) {
    stop("predicted and observed cannot be empty.")
  }

  if (anyNA(predicted) || anyNA(observed)) {
    stop("predicted and observed cannot contain missing values.")
  }

  if (!is.factor(predicted) || !is.factor(observed)) {
    stop("predicted and observed must be factors.")
  }

  if (!identical(levels(predicted), levels(observed))) {
    stop("predicted and observed must contain the same class levels.")
  }

  if (nlevels(observed) < 2) {
    stop("At least two class levels are required.")
  }

  if (
    length(n_boot) != 1 ||
    is.na(n_boot) ||
    n_boot < 0 ||
    n_boot != as.integer(n_boot)
  ) {
    stop("n_boot must be a non-negative integer.")
  }

  if (
    length(conf_level) != 1 ||
    is.na(conf_level) ||
    conf_level <= 0 ||
    conf_level >= 1
  ) {
    stop("conf_level must be between 0 and 1.")
  }

  if (
    !is.null(seed) &&
    (
      length(seed) != 1 ||
      is.na(seed) ||
      !is.numeric(seed)
    )
  ) {
    stop("seed must be NULL or a single numeric value.")
  }

  # Confusion matrix
  cm <- caret::confusionMatrix(
    predicted,
    observed
  )

  cm_tab <- table(
    Predicted = predicted,
    Observed = observed
  )

  # Point estimates
  point_results <- .classification_metric_values(
    predicted = predicted,
    observed = observed
  )

  values <- point_results$values
  class_metrics <- point_results$class_metrics

  metrics <- list(
    accuracy = unname(values["accuracy"]),
    kappa = unname(values["kappa"]),
    classwise = class_metrics,
    macro_precision = unname(values["macro_precision"]),
    macro_recall = unname(values["macro_recall"]),
    macro_specificity = unname(values["macro_specificity"]),
    macro_f1 = unname(values["macro_f1"])
  )

  # Bootstrap confidence intervals
  confidence_intervals <- NULL

  if (n_boot > 0) {

    # Preserve the current random state when seed is supplied
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

    bootstrap_values <- matrix(
      NA_real_,
      nrow = n_boot,
      ncol = length(values),
      dimnames = list(
        NULL,
        names(values)
      )
    )

    n <- length(observed)

    for (b in seq_len(n_boot)) {

      index <- sample(
        seq_len(n),
        size = n,
        replace = TRUE
      )

      bootstrap_results <- .classification_metric_values(
        predicted = predicted[index],
        observed = observed[index]
      )

      bootstrap_values[b, ] <- bootstrap_results$values
    }

    # Summarize confidence intervals
    alpha <- 1 - conf_level

    min_valid <- ceiling(
      0.80 * n_boot
    )

    ci_list <- lapply(
      seq_along(values),
      function(i) {

        x <- bootstrap_values[, i]
        x <- x[is.finite(x)]

        if (length(x) < min_valid) {
          lower <- NA_real_
          upper <- NA_real_
        } else {

          limits <- stats::quantile(
            x,
            probs = c(alpha / 2, 1 - alpha / 2),
            names = FALSE,
            na.rm = TRUE
          )

          lower <- limits[1]
          upper <- limits[2]
        }

        metric_name <- names(values)[i]

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

        data.frame(
          Metric = metric,
          Class = class,
          Estimate = unname(values[i]),
          Lower = lower,
          Upper = upper,
          N_valid = length(x),
          N_boot = n_boot,
          Method = "percentile_bootstrap",
          stringsAsFactors = FALSE
        )
      }
    )

    confidence_intervals <- do.call(
      rbind,
      ci_list
    )

    rownames(confidence_intervals) <- NULL
  }

  list(
    confusion_matrix = cm,
    confusion_table = cm_tab,
    metrics = metrics,
    confidence_intervals = confidence_intervals
  )
}

# =============================================================================
# 8. Classification ROC
# =============================================================================
#' @keywords internal

.classification_roc <- function(
    observed,
    probabilities,
    positive_class = NULL,
    conf_level = 0.95
) {

  # Validate observed
  if (!is.factor(observed)) {
    stop("observed must be a factor.")
  }

  if (nlevels(observed) != 2) {
    stop("ROC analysis requires exactly two classes.")
  }

  if (anyNA(observed)) {
    stop("observed cannot contain missing values.")
  }

  if (length(observed) == 0) {
    stop("observed cannot be empty.")
  }

  if (length(unique(observed)) != 2) {
    stop("Both classes must be represented in observed.")
  }

  # Validate confidence level
  if (
    length(conf_level) != 1 ||
    is.na(conf_level) ||
    conf_level <= 0 ||
    conf_level >= 1
  ) {
    stop("conf_level must be between 0 and 1.")
  }

  # Define positive class
  pos_class <- if (is.null(positive_class)) {
    levels(observed)[2]
  } else {
    positive_class
  }

  if (
    length(pos_class) != 1 ||
    is.na(pos_class) ||
    !pos_class %in% levels(observed)
  ) {
    stop("Invalid positive_class.")
  }

  # Validate probability matrix
  if (
    !is.matrix(probabilities) &&
    !is.data.frame(probabilities)
  ) {
    stop("probabilities must be a matrix or data.frame.")
  }

  if (nrow(probabilities) != length(observed)) {
    stop("probabilities and observed must have the same number of samples.")
  }

  if (
    !is.numeric(as.matrix(probabilities))
  ) {
    stop("Probabilities must contain numeric values.")
  }

  probabilities <- as.matrix(probabilities)

  if (
    is.null(colnames(probabilities)) ||
    anyDuplicated(colnames(probabilities))
  ) {
    stop("Probability columns must have unique class names.")
  }

  if (!pos_class %in% colnames(probabilities)) {
    stop("Positive class was not found among probability columns.")
  }

  if (
    anyNA(probabilities) ||
    any(!is.finite(probabilities))
  ) {
    stop("Predicted probabilities contain missing or non-finite values.")
  }

  if (
    any(probabilities < 0) ||
    any(probabilities > 1)
  ) {
    stop("Predicted probabilities must be between 0 and 1.")
  }

  # Define negative class
  neg_class <- setdiff(
    levels(observed),
    pos_class
  )

  # Calculate ROC
  roc_obj <- pROC::roc(
    response = observed,
    predictor = probabilities[, pos_class],
    levels = c(neg_class, pos_class),
    direction = "<",
    quiet = TRUE
  )

  # Calculate AUC
  auc_val <- as.numeric(
    pROC::auc(roc_obj)
  )

  # DeLong confidence interval
  auc_ci <- pROC::ci.auc(
    roc_obj,
    conf.level = conf_level,
    method = "delong"
  )

  list(
    roc = roc_obj,
    auc = auc_val,
    auc_ci = as.numeric(auc_ci),
    positive_class = pos_class,
    negative_class = neg_class
  )
}

# =============================================================================
# 9. Identify predictor types
# =============================================================================
#' @keywords internal

.get_predictor_types <- function(x) {

  vapply(
    x,
    function(z) {

      if (is.factor(z)) {
        "categorical"

      } else if (is.numeric(z)) {
        "numeric"

      } else {
        NA_character_
      }

    },
    character(1)
  )

}

# =============================================================================
# 10. Format metric CI
# =============================================================================
#' @keywords internal

.format_metric_ci <- function(
    estimate,
    ci,
    digits = 4
) {

  value <- format(
    round(estimate, digits),
    nsmall = digits
  )

  if (
    is.null(ci) ||
    length(ci) != 2 ||
    any(!is.finite(ci))
  ) {
    return(value)
  }

  lower <- format(round(ci[1], digits), nsmall = digits)
  upper <- format(round(ci[2], digits), nsmall = digits)

  paste0(
    value,
    " [",
    lower,
    "-",
    upper,
    "]"
  )
}

# =============================================================================
# 11. Get confidence intervals
# =============================================================================
#' @keywords internal

.get_ci <- function(
    confidence_intervals,
    metric
) {

  ci <- confidence_intervals

  if (
    is.null(ci) ||
    !is.data.frame(ci) ||
    !metric %in% ci$Metric
  ) {
    return(NULL)
  }

  row <- ci[
    ci$Metric == metric,
    ,
    drop = FALSE
  ]

  c(
    row$Lower[1],
    row$Upper[1]
  )
}
