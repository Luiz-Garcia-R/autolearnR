# =============================================================================
# Utils presentation
# =============================================================================

# =============================================================================
# 1. Print header
# =============================================================================
#' @keywords internal

.print_header <- function(title) {
  cat("\n")
  cat(strrep("=", 50), "\n")
  cat(title, "\n")
  cat(strrep("=", 50), "\n")
}

# =============================================================================
# 2. Print block
# =============================================================================
#' @keywords internal

.print_block <- function(title, content, width = 40) {
  cat("\n", title, "\n", sep = "")
  cat(strrep("-", width), "\n")
  content()
  cat(strrep("-", width), "\n")
}


# =============================================================================
# 3. Plot confusion matrix
# =============================================================================
#' @keywords internal

.plot_confusion_matrix <- function(
    confusion_table
) {

  # Convert confusion table to data.frame
  cm_df <- as.data.frame(confusion_table)

  if (ncol(cm_df) != 3) {
    stop(
      "confusion_table must contain two class dimensions and frequencies."
    )
  }

  # Standardize column names
  colnames(cm_df) <- c(
    "Predicted",
    "Observed",
    "Freq"
  )

  # Build plot
  ggplot2::ggplot(
    cm_df,
    ggplot2::aes(
      x = .data$Predicted,
      y = .data$Observed,
      fill = .data$Freq
    )
  ) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(
      ggplot2::aes(label = .data$Freq),
      color = "white",
      size = 5
    ) +
    ggplot2::scale_fill_gradient(
      low = "lightblue",
      high = "steelblue",
      guide = "none"
    ) +
    ggplot2::theme_bw(
      base_size = 12
    ) +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(
        angle = 45,
        hjust = 1
      )
    ) +
    ggplot2::labs(
      title = "Confusion Matrix",
      x = "Predicted",
      y = "Observed"
    )
}

# =============================================================================
# 4. Plot OOB error
# =============================================================================
#' @keywords internal

.plot_oob_error <- function(
    oob_trajectory
) {

  # Validate input
  if (
    !is.data.frame(oob_trajectory) ||
    !all(c("Tree", "OOB") %in% colnames(oob_trajectory))
  ) {
    stop(
      "oob_trajectory must be a data.frame containing ",
      "'Tree' and 'OOB' columns."
    )
  }

  if (nrow(oob_trajectory) == 0) {
    stop("oob_trajectory cannot be empty.")
  }

  if (
    !is.numeric(oob_trajectory$Tree) ||
    !is.numeric(oob_trajectory$OOB)
  ) {
    stop("'Tree' and 'OOB' columns must be numeric.")
  }

  # Build plot
  ggplot2::ggplot(
    oob_trajectory,
    ggplot2::aes(
      x = .data$Tree,
      y = .data$OOB
    )
  ) +
    ggplot2::geom_line(
      color = "steelblue",
      linewidth = 0.8
    ) +
    ggplot2::theme_bw(
      base_size = 12
    ) +
    ggplot2::labs(
      title = "Out-of-Bag Error",
      x = "Number of Trees",
      y = "OOB Error Rate"
    )
}

# =============================================================================
# 5. Plot feature importance
# =============================================================================
#' @keywords internal

.plot_feature_importance <- function(
    feature_importance,
    importance_col
) {

  ggplot2::ggplot(
    feature_importance,
    ggplot2::aes(
      x = stats::reorder(
        .data$Feature,
        .data[[importance_col]]
      ),
      y = .data[[importance_col]]
    )
  ) +
    ggplot2::geom_col(
      fill = "steelblue"
    ) +
    ggplot2::coord_flip() +
    ggplot2::theme_bw(
      base_size = 12
    ) +
    ggplot2::labs(
      title = "Feature Importance",
      x = NULL,
      y = "Importance"
    )
}

# =============================================================================
# 6. Plot ROC Curve
# =============================================================================
#' @keywords internal

.plot_roc_curve <- function(
    roc_object,
    auc_value
) {

  roc_df <- data.frame(
    FPR = rev(1 - roc_object$specificities),
    TPR = rev(roc_object$sensitivities)
  )

  ggplot2::ggplot(
    roc_df,
    ggplot2::aes(
      x = .data$FPR,
      y = .data$TPR
    )
  ) +
    ggplot2::geom_step(
      color = "steelblue",
      linewidth = 1,
      direction = "hv"
    ) +
    ggplot2::geom_abline(
      intercept = 0,
      slope = 1,
      linetype = "dashed",
      color = "gray50"
    ) +
    ggplot2::scale_x_continuous(
      limits = c(0, 1)
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1)
    ) +
    ggplot2::theme_bw(
      base_size = 12
    ) +
    ggplot2::labs(
      title = "ROC Curve",
      subtitle = paste0(
        "AUC = ",
        round(auc_value, 3)
      ),
      x = "1 - Specificity",
      y = "Sensitivity"
    )
}

# =============================================================================
# 7. Plot permutation test
# =============================================================================
#' @keywords internal

.plot_permutation_test <- function(
    permutation_accuracy,
    observed_accuracy,
    p_value
) {

  perm_df <- data.frame(
    Accuracy = permutation_accuracy
  )

  ggplot2::ggplot(
    perm_df,
    ggplot2::aes(x = .data$Accuracy)
  ) +
    ggplot2::geom_histogram(
      bins = 30,
      fill = "lightblue",
      color = "white"
    ) +
    ggplot2::geom_vline(
      xintercept = observed_accuracy,
      color = "red",
      linewidth = 1,
      linetype = "dashed"
    ) +
    ggplot2::theme_bw(
      base_size = 12
    ) +
    ggplot2::labs(
      title = "Permutation Test",
      subtitle = paste0(
        "Observed Accuracy = ",
        round(observed_accuracy, 3),
        " | p-value = ",
        signif(p_value, 3)
      ),
      x = "Accuracy under permutation",
      y = "Frequency"
    )
}

# =============================================================================
# 8. Combine classification plots
# =============================================================================
#' @keywords internal

.combine_classification_plots <- function(
    plots,
    ncol = 2
) {

  # Remove unavailable plots
  plots <- Filter(
    Negate(is.null),
    plots
  )

  if (length(plots) == 0) {
    return(NULL)
  }

  patchwork::wrap_plots(
    plots,
    ncol = ncol
  )
}

# =============================================================================
# 9. Plot decision tree
# =============================================================================
#' @keywords internal

.plot_decision_tree <- function(model) {

  if (!requireNamespace("rpart.plot", quietly = TRUE)) {
    stop(
      "The 'rpart.plot' package is required for decision tree visualization.\n",
      "Install using: install.packages('rpart.plot')",
      call. = FALSE
    )
  }

  rpart.plot::rpart.plot(
    model,
    type = 2,
    extra = 104,
    fallen.leaves = TRUE,
    nn = TRUE
  )

  invisible(model)
}

# =============================================================================
# 10. Plot cross-validation curve
# =============================================================================
#' @keywords internal

.plot_cv_curve <- function(
    cv_results,
    lambda_min,
    lambda_1se,
    lambda_choice
) {

  # Validate input
  required_cols <- c(
    "Lambda",
    "CV_Error",
    "CV_SD"
  )

  if (
    !is.data.frame(cv_results) ||
    !all(required_cols %in% colnames(cv_results))
  ) {
    stop(
      "cv_results must be a data.frame containing ",
      "'Lambda', 'CV_Error' and 'CV_SD' columns."
    )
  }

  if (nrow(cv_results) == 0) {
    stop("cv_results cannot be empty.")
  }

  if (!lambda_choice %in% c("min", "1se")) {
    stop("'lambda_choice' must be either 'min' or '1se'.")
  }

  # Build plot
  ggplot2::ggplot(
    cv_results,
    ggplot2::aes(
      x = log(.data$Lambda),
      y = .data$CV_Error
    )
  ) +
    ggplot2::geom_errorbar(
      ggplot2::aes(
        ymin = .data$CV_Error - .data$CV_SD,
        ymax = .data$CV_Error + .data$CV_SD
      ),
      width = 0.04,
      alpha = 0.5
    ) +
    ggplot2::geom_line(
      color = "steelblue",
      linewidth = 0.8
    ) +
    ggplot2::geom_point(
      color = "steelblue",
      size = 1.8
    ) +
    ggplot2::geom_vline(
      xintercept = log(lambda_min),
      linetype = "dashed",
      color = "steelblue"
    ) +
    ggplot2::geom_vline(
      xintercept = log(lambda_1se),
      linetype = "dashed",
      color = "firebrick"
    ) +
    ggplot2::theme_bw(
      base_size = 12
    ) +
    ggplot2::labs(
      title = "Cross-validation",
      subtitle = paste(
        "Selected lambda:",
        lambda_choice
      ),
      x = "Log(Lambda)",
      y = "Cross-validation error"
    )
}

# =============================================================================
# 11. Plot coefficient paths
# =============================================================================
#' @keywords internal

.plot_coefficient_paths <- function(
    model,
    model_family,
    selected_lambda,
    positive_class = NULL
) {

  # Extract coefficient trajectories
  beta_raw <- model$beta

  # Binary classification
  if (model_family == "binomial") {

    beta_matrix <- as.matrix(beta_raw)

    path_data <- data.frame(
      Feature = rep(
        rownames(beta_matrix),
        times = ncol(beta_matrix)
      ),
      Lambda = rep(
        model$lambda,
        each = nrow(beta_matrix)
      ),
      Coefficient = as.vector(beta_matrix),
      Class = positive_class,
      stringsAsFactors = FALSE
    )

  } else {

    # Multiclass classification
    path_list <- lapply(
      names(beta_raw),
      function(class_name) {

        beta_matrix <- as.matrix(
          beta_raw[[class_name]]
        )

        data.frame(
          Feature = rep(
            rownames(beta_matrix),
            times = ncol(beta_matrix)
          ),
          Lambda = rep(
            model$lambda,
            each = nrow(beta_matrix)
          ),
          Coefficient = as.vector(beta_matrix),
          Class = class_name,
          stringsAsFactors = FALSE
        )

      }
    )

    path_data <- do.call(
      rbind,
      path_list
    )

  }

  path_data$Log_Lambda <- log(
    path_data$Lambda
  )

  # Build plot
  p <- ggplot2::ggplot(
    path_data,
    ggplot2::aes(
      x = .data$Log_Lambda,
      y = .data$Coefficient,
      group = .data$Feature,
      color = .data$Feature
    )
  ) +
    ggplot2::geom_line(
      linewidth = 0.7,
      alpha = 0.8
    ) +
    ggplot2::geom_vline(
      xintercept = log(selected_lambda),
      linetype = "dashed",
      color = "black"
    ) +
    ggplot2::theme_bw(
      base_size = 12
    ) +
    ggplot2::labs(
      title = "Coefficient Paths",
      x = "Log(Lambda)",
      y = "Coefficient",
      color = "Feature"
    ) +
    ggplot2::theme(
      legend.position = "right",
      legend.title = ggplot2::element_text(
        face = "bold"
      ),
      legend.text = ggplot2::element_text(
        size = 8
      ),
      legend.key.height = grid::unit(
        0.35,
        "cm"
      )
    ) +
    ggplot2::guides(
      color = ggplot2::guide_legend(
        override.aes = list(
          linewidth = 1.2,
          alpha = 1
        )
      )
    )

  # Separate trajectories by class in multinomial models
  if (model_family == "multinomial") {

    p <- p +
      ggplot2::facet_wrap(
        ~ Class,
        scales = "free_y"
      )

  }

  p
}

# =============================================================================
# 12. SVM decision boundary
# =============================================================================
#' @keywords internal

.plot_svm_boundary <- function(
    model,
    x_train,
    y_train,
    preprocessor,
    kernel,
    grid_size = 200,
    expand = 0.05
) {

  # ===============================================
  # 12.1. Validate inputs
  # ===============================================

  if (ncol(x_train) != 2) {

    stop(
      "Decision boundary visualization requires exactly two predictors."
    )

  }

  if (length(y_train) != nrow(x_train)) {

    stop(
      "'y_train' must have the same number of observations as 'x_train'."
    )

  }

  if (
    !is.numeric(grid_size) ||
    length(grid_size) != 1 ||
    !is.finite(grid_size) ||
    grid_size < 2 ||
    grid_size != floor(grid_size)
  ) {

    stop("'grid_size' must be an integer greater than or equal to 2.")

  }

  if (
    !is.numeric(expand) ||
    length(expand) != 1 ||
    !is.finite(expand) ||
    expand < 0
  ) {

    stop("'expand' must be a non-negative finite number.")

  }

  feature_names <- colnames(x_train)

  if (is.null(feature_names)) {

    stop("'x_train' must have named columns.")

  }

  if (
    is.null(preprocessor$centers) ||
    is.null(preprocessor$scales)
  ) {

    stop(
      "The preprocessor must contain centers and scales."
    )

  }

  # ===============================================
  # 12.2. Recover original training coordinates
  # ===============================================

  if (isTRUE(preprocessor$scale)) {

    centers <- preprocessor$centers[feature_names]
    scales <- preprocessor$scales[feature_names]

    if (
      anyNA(centers) ||
      anyNA(scales) ||
      any(!is.finite(centers)) ||
      any(!is.finite(scales)) ||
      any(scales <= 0)
    ) {

      stop(
        "Invalid centers or scales in the SVM preprocessor."
      )

    }

  } else {

    centers <- setNames(
      rep(0, length(feature_names)),
      feature_names
    )

    scales <- setNames(
      rep(1, length(feature_names)),
      feature_names
    )

  }

  x_train_original <- sweep(
    as.matrix(x_train),
    2,
    scales,
    "*"
  )

  x_train_original <- sweep(
    x_train_original,
    2,
    centers,
    "+"
  )

  x_train_original <- as.data.frame(
    x_train_original
  )

  colnames(x_train_original) <- feature_names

  # ===============================================
  # 12.3. Define plotting grid in original coordinates
  # ===============================================

  feature_ranges <- lapply(
    x_train_original,
    range,
    na.rm = TRUE
  )

  grid_ranges <- lapply(
    feature_ranges,
    function(r) {

      padding <- diff(r) * expand

      if (padding == 0) {
        padding <- max(abs(r), 1) * expand
      }

      c(
        r[1] - padding,
        r[2] + padding
      )

    }
  )

  grid_original <- expand.grid(

    seq(
      grid_ranges[[1]][1],
      grid_ranges[[1]][2],
      length.out = grid_size
    ),

    seq(
      grid_ranges[[2]][1],
      grid_ranges[[2]][2],
      length.out = grid_size
    )

  )

  colnames(grid_original) <- feature_names

  # ===============================================
  # 12.4. Transform grid to model scale
  # ===============================================

  grid_processed <- sweep(
    as.matrix(grid_original),
    2,
    centers,
    "-"
  )

  grid_processed <- sweep(
    grid_processed,
    2,
    scales,
    "/"
  )

  colnames(grid_processed) <- feature_names

  # ===============================================
  # 12.5. Predict grid classes
  # ===============================================

  grid_predictions <- predict(
    model,
    newdata = grid_processed
  )

  grid_data <- data.frame(
    x = grid_original[[feature_names[1]]],
    y = grid_original[[feature_names[2]]],
    Predicted = factor(
      as.character(grid_predictions),
      levels = levels(y_train)
    )
  )

  # ===============================================
  # 12.6. Prepare training observations
  # ===============================================

  train_data <- data.frame(
    x = x_train_original[[feature_names[1]]],
    y = x_train_original[[feature_names[2]]],
    Class = y_train
  )

  # ===============================================
  # 12.7. Identify support vectors
  # ===============================================

  sv_idx <- model$index

  if (is.null(sv_idx)) {
    sv_idx <- integer(0)
  }

  sv_idx <- unique(
    as.integer(sv_idx)
  )

  sv_idx <- sv_idx[
    !is.na(sv_idx) &
      sv_idx >= 1 &
      sv_idx <= nrow(train_data)
  ]

  sv_data <- train_data[
    sv_idx,
    ,
    drop = FALSE
  ]

  # ===============================================
  # 12.8. Build decision boundary plot
  # ===============================================

  p <- ggplot2::ggplot() +

    # Predicted decision regions
    ggplot2::geom_raster(
      data = grid_data,
      ggplot2::aes(
        x = .data$x,
        y = .data$y,
        fill = .data$Predicted
      ),
      alpha = 0.25
    ) +

    # Training observations
    ggplot2::geom_point(
      data = train_data,
      ggplot2::aes(
        x = .data$x,
        y = .data$y,
        color = .data$Class
      ),
      size = 2.5,
      alpha = 0.85
    ) +

    # Support vectors
    ggplot2::geom_point(
      data = sv_data,
      ggplot2::aes(
        x = .data$x,
        y = .data$y
      ),
      shape = 21,
      fill = NA,
      color = "black",
      size = 4,
      stroke = 1
    ) +

    ggplot2::labs(
      title = "SVM Decision Boundary",
      subtitle = paste(
        "Kernel:",
        kernel,
        "| Support vectors:",
        length(sv_idx)
      ),
      x = feature_names[1],
      y = feature_names[2],
      fill = "Predicted region",
      color = "Training class",
    ) +

    ggplot2::theme_minimal() +

    ggplot2::theme(
      plot.title = ggplot2::element_text(
      ),
      legend.position = "right"
    )

  return(p)

}

# =============================================================================
# Compare Plot Palette
# =============================================================================
#' @keywords internal

.compare_palette <- function(algorithms) {

  base_colors <- c(
    tree = "#332288",
    logistic = "#117733",
    elastic = "#44AA99",
    rf = "#88CCEE",
    svm = "#DDCC77"
  )

  # Match colors to algorithm names
  available_colors <- base_colors[
    match(tolower(algorithms), names(base_colors))
  ]

  # Handle unknown algorithms
  missing_colors <- is.na(available_colors)

  if (any(missing_colors)) {
    extra_colors <- grDevices::hcl.colors(
      sum(missing_colors),
      palette = "Dark 3"
    )

    available_colors[missing_colors] <- extra_colors
  }

  names(available_colors) <- algorithms

  available_colors
}

# =============================================================================
# 14. Plot ROC Comparison
# =============================================================================
#' @keywords internal

.plot_compare_roc <- function(roc_data) {

  # Validate input
  if (
    is.null(roc_data$curves) ||
    is.null(roc_data$summary)
  ) {
    stop(
      "'roc_data' must contain 'curves' and 'summary'."
    )
  }

  required_curve_cols <- c(
    "Algorithm", "FPR", "TPR"
  )

  required_summary_cols <- c(
    "Algorithm", "AUC"
  )

  if (!all(required_curve_cols %in% names(roc_data$curves))) {
    stop(
      "'curves' is missing required columns."
    )
  }

  if (!all(required_summary_cols %in% names(roc_data$summary))) {
    stop(
      "'summary' is missing required columns."
    )
  }

  # Validate algorithm consistency
  algorithms <- roc_data$summary$Algorithm

  if (
    anyDuplicated(algorithms) ||
    !setequal(
      algorithms,
      unique(roc_data$curves$Algorithm)
    )
  ) {
    stop(
      "Algorithms in 'curves' and 'summary' must match."
    )
  }

  # Set colors
  algorithm_colors <- .compare_palette(algorithms)

  # Prepare AUC labels
  auc_labels <- setNames(
    sprintf(
      "%s (AUC = %.3f)",
      algorithms,
      roc_data$summary$AUC
    ),
    algorithms
  )

  # Prepare curve data
  plot_data <- roc_data$curves

  # Set algorithm ordering
  plot_data$Algorithm <- factor(
    plot_data$Algorithm,
    levels = algorithms
  )

  # Order ROC coordinates within each algorithm
  plot_data <- do.call(
    rbind,
    lapply(
      algorithms,
      function(algorithm) {

        algorithm_data <- plot_data[
          plot_data$Algorithm == algorithm,
          ,
          drop = FALSE
        ]

        algorithm_data <- algorithm_data[
          order(
            algorithm_data$FPR,
            algorithm_data$TPR
          ),
          ,
          drop = FALSE
        ]

        algorithm_data
      }
    )
  )

  # Build ROC comparison plot
  ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = .data$FPR,
      y = .data$TPR,
      color = .data$Algorithm,
      group = .data$Algorithm
    )
  ) +

    # ROC curves
    ggplot2::geom_step(
      linewidth = 1,
      direction = "hv"
    ) +

    # Random classifier reference
    ggplot2::geom_abline(
      intercept = 0,
      slope = 1,
      linetype = "dashed",
      color = "gray50"
    ) +

    # Axis limits
    ggplot2::scale_x_continuous(
      limits = c(0, 1),
      expand = ggplot2::expansion(mult = 0.05)
    ) +

    ggplot2::scale_y_continuous(
      limits = c(0, 1),
      expand = ggplot2::expansion(mult = 0.05)
    ) +

    # Algorithm colors and AUC labels
    ggplot2::scale_color_manual(
      values = algorithm_colors,
      breaks = algorithms,
      labels = unname(auc_labels)
    ) +

    # Plot theme
    ggplot2::theme_bw(
      base_size = 12
    ) +

    # Legend
    ggplot2::theme(
      legend.position = c(0.98, 0.02),
      legend.justification = c(1, 0),
      legend.background = ggplot2::element_rect(
        fill = "white",
        color = "gray70"
      ),
      legend.key = ggplot2::element_blank()
    ) +

    # Labels
    ggplot2::labs(
      title = "ROC Curve Comparison",
      x = "1 - Specificity",
      y = "Sensitivity",
      color = NULL
    ) +

    # Square plotting area
    ggplot2::coord_equal()
}

# =============================================================================
# 15. Plot Confusion Matrix Comparison
# =============================================================================
#' @keywords internal

.plot_compare_confusion <- function(compare_data) {

  # Validate input
  if (is.null(compare_data$predictions)) {
    stop(
      "'compare_data' must contain 'predictions'."
    )
  }

  predictions <- compare_data$predictions

  required_cols <- c(
    "Sample", "Observed", "Predicted", "Algorithm"
  )

  if (!all(required_cols %in% names(predictions))) {
    stop(
      "'predictions' is missing required columns."
    )
  }

  # Validate class information
  class_levels <- levels(predictions$Observed)

  if (
    is.null(class_levels) ||
    length(class_levels) < 2
  ) {
    stop(
      "'Observed' must be a factor with at least two class levels."
    )
  }

  if (
    anyNA(predictions$Observed) ||
    anyNA(predictions$Predicted) ||
    anyNA(predictions$Algorithm)
  ) {
    stop(
      "'predictions' contains missing values."
    )
  }

  if (!all(
    as.character(predictions$Predicted) %in% class_levels
  )) {
    stop(
      "'Predicted' contains classes not present in 'Observed'."
    )
  }

  # Define algorithm and class ordering
  algorithms <- unique(predictions$Algorithm)

  predictions$Algorithm <- factor(
    predictions$Algorithm,
    levels = algorithms
  )

  predictions$Observed <- factor(
    predictions$Observed,
    levels = class_levels
  )

  predictions$Predicted <- factor(
    predictions$Predicted,
    levels = class_levels
  )

  # Calculate confusion matrix counts
  confusion_data <- as.data.frame(
    table(
      Algorithm = predictions$Algorithm,
      Observed = predictions$Observed,
      Predicted = predictions$Predicted
    )
  )

  names(confusion_data)[4] <- "Freq"

  # Common color scale
  max_count <- max(confusion_data$Freq)

  # Build plot
  ggplot2::ggplot(
    confusion_data,
    ggplot2::aes(
      x = .data$Predicted,
      y = .data$Observed,
      fill = .data$Freq
    )
  ) +

    ggplot2::geom_tile() +

    ggplot2::geom_text(
      ggplot2::aes(
        label = .data$Freq
      ),
      color = "white",
      size = 5
    ) +

    ggplot2::facet_wrap(
      ~Algorithm
    ) +

    ggplot2::scale_fill_gradient(
      low = "lightblue",
      high = "steelblue",
      limits = c(0, max_count),
      guide = "none"
    ) +

    ggplot2::scale_x_discrete(
      limits = class_levels
    ) +

    ggplot2::scale_y_discrete(
      limits = rev(class_levels)
    ) +

    ggplot2::coord_equal() +

    ggplot2::theme_bw(
      base_size = 12
    ) +

    ggplot2::theme(
      axis.text.x = ggplot2::element_text(
        angle = 45,
        hjust = 1
      ),
      panel.grid = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(
        face = "bold"
      )
    ) +

    ggplot2::labs(
      title = "Confusion Matrix Comparison",
      x = NULL,
      y = NULL
    )
}

# =============================================================================
# 16. Plot Performance Comparison
# =============================================================================
#' @keywords internal


.plot_compare_performance <- function(compare_metrics) {

  # ===============================================
  # 1. Validate input
  # ===============================================

  if (
    is.null(compare_metrics$summary) ||
    !is.data.frame(compare_metrics$summary)
  ) {
    stop(
      "'compare_metrics' must contain a 'summary' data.frame."
    )
  }

  metrics_df <- compare_metrics$summary

  # ===============================================
  # 2. Validate required columns
  # ===============================================

  metric_cols <- c(
    "Accuracy",
    "Balanced_Accuracy",
    "Kappa",
    "Precision",
    "Recall",
    "F1"
  )

  required_cols <- c(
    "Algorithm",
    metric_cols
  )

  if (!all(required_cols %in% names(metrics_df))) {
    stop(
      "'summary' is missing required metric columns."
    )
  }

  if (!all(vapply(
    metrics_df[metric_cols],
    is.numeric,
    logical(1)
  ))) {
    stop(
      "All performance metrics must be numeric."
    )
  }

  if (any(!is.finite(
    as.matrix(metrics_df[metric_cols])
  ))) {
    stop(
      "Performance metrics contain non-finite values."
    )
  }

  # ===============================================
  # 3. Convert metrics and confidence intervals
  #    to long format
  # ===============================================

  plot_data <- do.call(
    rbind,
    lapply(metric_cols, function(metric) {

      lower_col <- paste0(metric, "_Lower")
      upper_col <- paste0(metric, "_Upper")

      lower <- if (lower_col %in% names(metrics_df)) {
        metrics_df[[lower_col]]
      } else {
        rep(NA_real_, nrow(metrics_df))
      }

      upper <- if (upper_col %in% names(metrics_df)) {
        metrics_df[[upper_col]]
      } else {
        rep(NA_real_, nrow(metrics_df))
      }

      data.frame(
        Algorithm = metrics_df$Algorithm,
        Metric = metric,
        Value = metrics_df[[metric]],
        Lower = lower,
        Upper = upper,
        stringsAsFactors = FALSE
      )

    })
  )

  rownames(plot_data) <- NULL

  # ===============================================
  # 4. Define metric ordering
  # ===============================================

  metric_labels <- c(
    Accuracy = "Accuracy",
    Balanced_Accuracy = "Balanced Accuracy",
    Kappa = "Kappa",
    Precision = "Precision",
    Recall = "Recall",
    F1 = "F1 Score"
  )

  plot_data$Metric <- factor(
    plot_data$Metric,
    levels = metric_cols,
    labels = unname(metric_labels)
  )

  plot_data$Algorithm <- factor(
    plot_data$Algorithm,
    levels = metrics_df$Algorithm
  )

  # ===============================================
  # 5. Identify available confidence intervals
  # ===============================================

  ci_data <- plot_data[
    is.finite(plot_data$Lower) &
      is.finite(plot_data$Upper),
    ,
    drop = FALSE
  ]

  # ===============================================
  # 6. Set colors
  # ===============================================

  algorithm_colors <- .compare_palette(
    metrics_df$Algorithm
  )

  # ===============================================
  # 7. Build performance comparison plot
  # ===============================================

  ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = .data$Algorithm,
      y = .data$Value,
      color = .data$Algorithm
    )
  ) +

    # Confidence intervals
    ggplot2::geom_errorbar(
      data = ci_data,
      ggplot2::aes(
        ymin = .data$Lower,
        ymax = .data$Upper
      ),
      width = 0.15,
      linewidth = 0.7
    ) +

    # Point estimates
    ggplot2::geom_point(
      size = 2.8
    ) +

    # Manual color scale
    ggplot2::scale_color_manual(
      values = algorithm_colors
    ) +

    # Separate metric panels
    ggplot2::facet_wrap(
      ~Metric,
      scales = "free_y",
      ncol = 2
    ) +

    # Horizontal confidence intervals
    ggplot2::coord_flip() +

    # Plot theme
    ggplot2::theme_bw(
      base_size = 12
    ) +

    ggplot2::theme(
      axis.text.x = ggplot2::element_text(
        color = "black"
      ),
      axis.text.y = ggplot2::element_text(
        color = "black"
      ),
      panel.grid.major.y = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(
        face = "bold"
      ),
      legend.position = "none"
    ) +

    # Labels
    ggplot2::labs(
      title = "Model Performance Comparison",
      x = NULL,
      y = "Metric Value"
    )
}


# =============================================================================
# 17. Plot prediction ROC comparison
# =============================================================================
#' @keywords internal

.plot_predict_roc <- function(
    reference_roc,
    external_roc
) {

  if (
    !inherits(reference_roc, "roc") ||
    !inherits(external_roc, "roc")
  ) {
    stop(
      "'reference_roc' and 'external_roc' must be pROC ROC objects."
    )
  }

  # Extract ROC coordinates
  reference_coords <- as.data.frame(
    pROC::coords(
      reference_roc,
      x = "all",
      ret = c("specificity", "sensitivity"),
      transpose = FALSE
    )
  )

  external_coords <- as.data.frame(
    pROC::coords(
      external_roc,
      x = "all",
      ret = c("specificity", "sensitivity"),
      transpose = FALSE
    )
  )

  # Standardize coordinates
  reference_data <- data.frame(
    FPR = 1 - reference_coords$specificity,
    TPR = reference_coords$sensitivity,
    Cohort = "Reference"
  )

  external_data <- data.frame(
    FPR = 1 - external_coords$specificity,
    TPR = external_coords$sensitivity,
    Cohort = "External"
  )

  plot_data <- rbind(
    reference_data,
    external_data
  )

  # Remove non-finite coordinates
  plot_data <- plot_data[
    is.finite(plot_data$FPR) &
      is.finite(plot_data$TPR),
    ,
    drop = FALSE
  ]

  # Order coordinates
  plot_data <- do.call(
    rbind,
    lapply(
      c("Reference", "External"),
      function(cohort) {

        cohort_data <- plot_data[
          plot_data$Cohort == cohort,
          ,
          drop = FALSE
        ]

        cohort_data[
          order(
            cohort_data$FPR,
            cohort_data$TPR
          ),
          ,
          drop = FALSE
        ]

      }
    )
  )

  # AUC labels
  auc_values <- c(
    Reference = as.numeric(pROC::auc(reference_roc)),
    External = as.numeric(pROC::auc(external_roc))
  )

  auc_labels <- sprintf(
    "%s (AUC = %.3f)",
    names(auc_values),
    auc_values
  )

  # Build plot
  ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = .data$FPR,
      y = .data$TPR,
      color = .data$Cohort,
      group = .data$Cohort
    )
  ) +

    ggplot2::geom_step(
      linewidth = 1,
      direction = "hv"
    ) +

    ggplot2::geom_abline(
      intercept = 0,
      slope = 1,
      linetype = "dashed",
      color = "gray50"
    ) +

    ggplot2::scale_x_continuous(
      limits = c(0, 1),
      expand = ggplot2::expansion(mult = 0.05)
    ) +

    ggplot2::scale_y_continuous(
      limits = c(0, 1),
      expand = ggplot2::expansion(mult = 0.05)
    ) +

    ggplot2::scale_color_manual(
      values = c(
        Reference = "gray40",
        External = "steelblue"
      ),
      breaks = names(auc_values),
      labels = auc_labels
    ) +

    ggplot2::theme_bw(
      base_size = 12
    ) +

    ggplot2::theme(
      legend.position = c(0.98, 0.02),
      legend.justification = c(1, 0),
      legend.background = ggplot2::element_rect(
        fill = "white",
        color = "gray70"
      ),
      legend.key = ggplot2::element_blank()
    ) +

    ggplot2::labs(
      title = "ROC Curve Comparison",
      x = "1 - Specificity",
      y = "Sensitivity",
      color = NULL
    ) +

    ggplot2::coord_equal()

}

# =============================================================================
# 18. Plot prediction performance forest plot
# =============================================================================
#' @keywords internal

.plot_predict_forest <- function(
    comparison_summary
) {

  required_cols <- c(
    "Metric",
    "Class",
    "Difference",
    "Lower",
    "Upper"
  )

  if (
    !is.data.frame(comparison_summary) ||
    !all(required_cols %in% names(comparison_summary))
  ) {
    stop(
      "'comparison_summary' is missing required columns."
    )
  }

  if (nrow(comparison_summary) == 0) {
    stop(
      "'comparison_summary' must contain at least one comparison."
    )
  }

  plot_data <- comparison_summary

  # Metric labels
  metric_labels <- c(
    accuracy = "Accuracy",
    kappa = "Kappa",
    macro_precision = "Macro precision",
    macro_recall = "Macro recall",
    macro_f1 = "Macro F1",
    precision = "Precision",
    recall = "Recall",
    f1 = "F1",
    auc = "AUC"
  )

  plot_data$Metric_label <- unname(
    metric_labels[plot_data$Metric]
  )

  # Add class information
  plot_data$Label <- ifelse(
    is.na(plot_data$Class),
    plot_data$Metric_label,
    paste(
      plot_data$Metric_label,
      plot_data$Class,
      sep = " - "
    )
  )

  # Define display order
  global_metrics <- c(
    "Accuracy",
    "Kappa",
    "Macro precision",
    "Macro recall",
    "Macro F1"
  )

  class_metrics <- c(
    "Precision",
    "Recall",
    "F1"
  )

  class_levels <- unique(
    plot_data$Class[!is.na(plot_data$Class)]
  )

  metric_order <- c(
    global_metrics,
    unlist(
      lapply(
        class_metrics,
        function(metric) {
          paste(metric, class_levels, sep = " - ")
        }
      )
    ),
    if ("auc" %in% plot_data$Metric) "AUC"
  )

  metric_order <- metric_order[
    metric_order %in% plot_data$Label
  ]

  plot_data$Label <- factor(
    plot_data$Label,
    levels = rev(metric_order)
  )

  # Build plot
  ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = .data$Difference,
      y = .data$Label
    )
  ) +

    # No difference reference
    ggplot2::geom_vline(
      xintercept = 0,
      linetype = "dashed",
      color = "gray50"
    ) +

    # Confidence intervals
    ggplot2::geom_errorbar(
      data = plot_data[
        is.finite(plot_data$Lower) &
          is.finite(plot_data$Upper),
        ,
        drop = FALSE
      ],
      ggplot2::aes(
        xmin = .data$Lower,
        xmax = .data$Upper
      ),
      orientation = "y",
      width = 0.2,
      linewidth = 0.7,
      color = "steelblue"
    ) +

    # Point estimates
    ggplot2::geom_point(
      size = 2.8,
      color = "steelblue"
    ) +

    ggplot2::theme_bw(
      base_size = 12
    ) +

    ggplot2::theme(
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      axis.text.y = ggplot2::element_text(
        color = "black"
      )
    ) +

    ggplot2::labs(
      title = "Generalization Performance",
      subtitle = "Difference in performance (External - Reference)",
      x = "Difference with 95% confidence interval",
      y = NULL
    )

}

