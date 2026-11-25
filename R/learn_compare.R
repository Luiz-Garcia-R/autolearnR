# =============================================================================
# LEARN COMPARE
# =============================================================================

#' Compare Classification Models
#'
#' Compares the classification performance of multiple models previously trained
#' using the \code{learn_*} functions. The function evaluates predictive
#' performance using a common test set, calculates aggregated and class-wise
#' performance metrics, and generates comparative visualizations.
#'
#' Performance metrics include Accuracy, Kappa, Balanced Accuracy, Precision,
#' Recall, Specificity, and F1-score. Metrics can be summarized using macro- or
#' weighted averaging, while class-wise metrics are retained for individual
#' class assessment.
#'
#' Confidence intervals are retrieved from the individual model objects and
#' incorporated into the comparative results. The function does not recalculate
#' confidence intervals or perform statistical tests of differences between
#' algorithms.
#'
#' For binary classification, comparative ROC curves and AUC values can also be
#' calculated using predicted class probabilities.
#'
#' Three comparative plots are available: a performance comparison, ROC curve
#' comparison, and confusion matrix comparison.
#'
#' @param results A named list containing classification model objects returned
#'   by supported \code{learn_*} functions. Currently supported algorithms
#'   include \code{learn_tree}, \code{learn_logistic}, \code{learn_elastic_net},
#'   \code{learn_random_forest}, and \code{learn_svm}. Each list element must have a
#'   unique, non-empty name identifying the corresponding algorithm.
#'
#' @param average Character string specifying the averaging method used to
#'   calculate aggregated performance metrics. Defaults to \code{"macro"}. Macro
#'   averaging assigns equal weight to each class, whereas weighted averaging
#'   accounts for class support.
#'
#' @param positive_class Optional character string specifying the positive class
#'   used for ROC analysis. If \code{NULL}, the positive class stored in the
#'   input model objects is used, provided that all models share the same
#'   positive class. An explicitly supplied value overrides the positive class
#'   stored in the individual models.
#'
#' @param roc Logical value indicating whether comparative ROC curves and AUC
#'   values should be calculated. Defaults to \code{TRUE}. ROC analysis is
#'   restricted to binary classification and requires probability estimates from
#'   all compared models. If these conditions are not met, ROC analysis is
#'   skipped with a warning.
#'
#' @param plot Logical value indicating whether comparative plots should be
#'   generated and displayed. Defaults to \code{TRUE}. When \code{FALSE},
#'   numerical results are still calculated and returned, but no plots are
#'   generated.
#'
#' @return An object of class \code{learn_compare}, containing:
#' \describe{
#'   \item{call}{The function call used to generate the comparison.}
#'   \item{algorithms}{Character vector containing the names of the
#'   compared algorithms.}
#'   \item{n_models}{Number of models included in the comparison.}
#'   \item{n_train}{Named integer vector containing the number of
#'   training samples used by each model.}
#'   \item{n_test}{Named integer vector containing the number of test samples
#'   used by each model.}
#'   \item{classes}{Character vector containing the classification
#'   target levels.}
#'   \item{average}{Averaging method used for aggregated metrics.}
#'   \item{positive_class}{Positive class used for ROC analysis,
#'   or \code{NULL} if ROC analysis was not performed.}
#'   \item{metrics}{List containing aggregated and class-wise
#'   classification performance metrics:
#'     \describe{
#'       \item{summary}{Data frame containing Accuracy, Kappa,
#''       Balanced Accuracy, Precision, Recall, Specificity, and
#'       F1-score estimates and their available confidence interval
#'       bounds for each algorithm.}
#'       \item{classwise}{Data frame containing class-specific
#'       performance metrics, class support, and available
#'       confidence interval bounds for each algorithm.}
#'       \item{confidence_intervals}{Combined data frame containing
#'       the original confidence interval estimates retrieved from
#'       the individual model objects, including their methods and
#'       associated metadata.}
#'     }
#'   }
#'   \item{roc}{List containing comparative ROC results, when available:
#'     \describe{
#'       \item{summary}{Data frame containing AUC values for each algorithm.}
#'       \item{curves}{Data frame containing ROC coordinates,
#'       including thresholds, specificity, sensitivity, false
#'       positive rates, and true positive rates.}
#'     }
#'   }
#'   \item{predictions}{Combined data frame containing sample identifiers,
#'   observed classes, predicted classes, and algorithm names.}
#'   \item{probabilities}{Combined data frame containing sample
#'   identifiers, algorithm names, and predicted class probabilities,
#'   when available.}
#'   \item{compatibility}{List containing the compatibility report:
#'     \describe{
#'       \item{checks}{Data frame describing compatibility checks,
#'       their status, and corresponding messages.}
#'       \item{comparable}{Logical value indicating whether the
#'       models are suitable for comparison.}
#'       \item{roc_compatible}{Logical value indicating whether
#'       the models meet the compatibility requirements for ROC
#'       analysis.}
#'     }
#'   }
#'   \item{plots}{List containing comparative visualizations:
#'     \describe{
#'       \item{performance}{Comparative performance plot displaying
#'       Accuracy, Balanced Accuracy, Kappa, Precision, Recall, and F1-score,
#'       with available confidence intervals.}
#'       \item{roc}{Comparative ROC plot, when available and
#'       \code{plot = TRUE}.}
#'       \item{confusion}{Comparative confusion matrix plot, when
#'       \code{plot = TRUE}.}
#'     }
#'   }
#' }
#'
#' @details
#' The function does not train, optimize, or select classification models.
#' Instead, it evaluates and compares models previously fitted using
#' supported \code{learn_*} functions.
#'
#' Before calculating performance metrics, model compatibility is assessed
#' based on the target variable, class levels and ordering, observed test
#' labels, and sample identifiers. Differences in training or test sample
#' sets generate compatibility warnings when the remaining requirements
#' allow comparison. Models with incompatible target variables, class
#' levels, or observed test labels cannot be compared.
#'
#' Aggregated metrics are calculated using the selected averaging method.
#' Macro averaging gives equal importance to each class, whereas weighted
#' averaging accounts for the number of observations in each class.
#' Class-wise metrics are also retained to facilitate detailed assessment
#' of classification performance.
#'
#' Confidence intervals are retrieved from the individual model objects
#' and matched to their corresponding performance metrics. They are not
#' recalculated by \code{learn_compare}. Confidence intervals are included
#' only when the corresponding metric and averaging method are available
#' in the original model results.
#'
#' The function does not perform pairwise statistical tests or calculate
#' differences between algorithm performance estimates. Confidence
#' intervals should therefore be interpreted as individual measures of
#' estimation uncertainty, rather than as evidence of statistically
#' significant differences between algorithms.
#'
#' ROC analysis is restricted to binary classification and requires predicted
#' class probabilities from every compared model.
#' When \code{positive_class = NULL}, ROC analysis uses the common positive
#' class stored in the individual model objects. If the models do not share a
#' positive class, an explicit \code{positive_class} must be provided.
#'
#' The performance comparison plot displays Accuracy, Balanced Accuracy,
#' Kappa, Precision, Recall, and F1-score in separate panels.
#' Point estimates are displayed for all metrics, with confidence intervals
#' shown when available.
#'
#' The confusion matrix comparison displays absolute classification counts for
#' each algorithm. Comparisons using different test sample sets should
#' therefore be interpreted with caution.
#'
#' @examples
#' # Example: Compare classification algorithms using the iris dataset.
#'
#' # Select versicolor and virginica species
#' iris_data <- iris[
#'   iris$Species %in% c("versicolor", "virginica"),
#' ]
#'
#' # Prepare sample identifiers and target variable
#' iris_data$Sample <- rownames(iris_data)
#' iris_data$Species <- droplevels(iris_data$Species)
#'
#' # Define common model arguments
#' args <- list(
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
#'   positive_class = NULL,
#'   seed = 123,
#'   plot = FALSE
#' )
#'
#' # Define classification algorithms
#' functions <- list(
#'   tree = learn_tree,
#'   logistic = learn_logistic,
#'   elastic = learn_elastic_net,
#'   rf = learn_random_forest,
#'   svm = learn_svm
#' )
#'
#' # Train all models using the same dataset and settings
#' results <- lapply(
#'   functions,
#'   function(f) do.call(f, args)
#' )
#'
#' # Compare classification performance
#' comparison <- learn_compare(
#'   results = results,
#'   average = "macro",
#'   positive_class = "virginica",
#'   roc = TRUE,
#'   plot = TRUE
#' )
#'
#' @references
#' Sokolova, M., & Lapalme, G. (2009).
#' A systematic analysis of performance measures for classification tasks.
#' \emph{Information Processing & Management}, 45(4), 427-437.
#' https://doi.org/10.1016/j.ipm.2009.03.002
#'
#' Hand, D. J., & Till, R. J. (2001).
#' A simple generalisation of the area under the ROC curve for multiple
#' class classification problems.
#' \emph{Machine Learning}, 45, 171-186.
#' https://doi.org/10.1023/A:1010920819831
#'
#' @seealso
#' \code{\link{learn_tree}},
#' \code{\link{learn_logistic}},
#' \code{\link{learn_elastic_net}},
#' \code{\link{learn_random_forest}},
#' \code{\link{learn_svm}}
#'
#' @export

learn_compare <- function(
    results,
    average = "macro",
    positive_class = NULL,
    roc = TRUE,
    plot = TRUE
) {

# =============================================================================
# 1. Validate input
# =============================================================================

.validate_compare_input(
  results = results
)

# =============================================================================
# 2. Check model compatibility
# =============================================================================

compatibility <- .check_compare_compatibility(
  results = results
)

# Stop if models are not comparable
if (!compatibility$comparable) {

  incompatible <- compatibility$checks[
    compatibility$checks$status == "incompatible",
    ,
    drop = FALSE
  ]

  stop(
    "Models are not compatible for comparison: ",
    paste(incompatible$check, collapse = ", "),
    "."
  )
}

# =============================================================================
# 3. Extract comparison data
# =============================================================================

compare_data <- .extract_compare_data(
  results = results
)

# =============================================================================
# 4. Calculate performance metrics
# =============================================================================

compare_metrics <- .calculate_compare_metrics(
  compare_data = compare_data,
  average = average
)

# Attach existing confidence intervals
compare_metrics <- .attach_compare_confidence_intervals(
  compare_metrics = compare_metrics,
  confidence_intervals = compare_data$confidence_intervals,
  average = average
)

# =============================================================================
# 5. Calculate ROC
# =============================================================================

roc_data <- NULL
selected_positive_class <- NULL

if (roc) {

  class_levels <- levels(
    compare_data$predictions$Observed
  )

  # ROC requires binary classification
  if (length(class_levels) != 2) {

    warning(
      "ROC comparison is only available for binary classification. ",
      "ROC analysis was skipped.",
      call. = FALSE
    )

  } else if (is.null(compare_data$probabilities)) {

    warning(
      "Probability estimates are unavailable. ",
      "ROC analysis was skipped.",
      call. = FALSE
    )

  } else {

    # Determine positive class
    if (is.null(positive_class)) {

      model_positive_classes <- vapply(
        results,
        function(x) {

          value <- x$parameters$positive_class

          if (is.null(value) || length(value) != 1) {
            return(NA_character_)
          }

          as.character(value)
        },
        character(1)
      )

      if (
        anyNA(model_positive_classes) ||
        length(unique(model_positive_classes)) != 1
      ) {

        warning(
          "Models do not share a common positive class. ",
          "Please provide 'positive_class' explicitly. ",
          "ROC analysis was skipped.",
          call. = FALSE
        )

      } else {

        selected_positive_class <- model_positive_classes[1]

      }

    } else {

      if (
        length(positive_class) != 1 ||
        is.na(positive_class) ||
        !positive_class %in% class_levels
      ) {
        stop(
          "'positive_class' must match one of the observed class levels."
        )
      }

      selected_positive_class <- positive_class

    }

    # Calculate ROC when a valid positive class is available
    if (!is.null(selected_positive_class)) {

      probability_algorithms <- unique(
        compare_data$probabilities$Algorithm
      )

      if (!setequal(
        probability_algorithms,
        names(results)
      )) {

        warning(
          "Not all models provide probability estimates. ",
          "ROC analysis was skipped.",
          call. = FALSE
        )

        selected_positive_class <- NULL

      } else {

        roc_data <- .calculate_compare_roc(
          compare_data = compare_data,
          positive_class = selected_positive_class
        )

      }
    }
  }
}

# =============================================================================
# 6. Generate plots
# =============================================================================

plot_list <- list(
  performance = NULL,
  roc = NULL,
  confusion = NULL
)

if (plot) {

  # Performance comparison
  plot_list$performance <- .plot_compare_performance(
    compare_metrics
  )

  # ROC comparison
  if (!is.null(roc_data)) {

    plot_list$roc <- .plot_compare_roc(
      roc_data
    )

  }

  # Confusion matrix comparison
  plot_list$confusion <- .plot_compare_confusion(
    compare_data
  )

  # Display plots
  for (plot_object in plot_list) {

    if (!is.null(plot_object)) {
      print(plot_object)
    }
  }
}

# =============================================================================
# 7. Build output object
# =============================================================================

obj <- list(
  call = match.call(),
  algorithms = names(results),
  n_models = length(results),
  n_train = vapply(
    results,
    function(x) length(x$train_samples),
    integer(1)
  ),
  n_test = vapply(
    results,
    function(x) length(x$test_samples),
    integer(1)
  ),
  classes = levels(
    compare_data$predictions$Observed
  ),
  average = average,
  positive_class = selected_positive_class,
  metrics = compare_metrics,
  roc = roc_data,
  predictions = compare_data$predictions,
  probabilities = compare_data$probabilities,
  compatibility = compatibility,
  plots = plot_list
)

class(obj) <- "learn_compare"

# =============================================================================
# 8. Print results
# =============================================================================

print(obj)

return(invisible(obj))
}

# =============================================================================
# Print S3
# =============================================================================
#' Print method for learn_compare objects
#'
#' Displays a formatted summary of a \code{learn_compare} object.
#'
#' @param x An object of class \code{learn_compare}.
#' @param ... Further arguments (ignored).
#' @return Invisibly returns \code{x}.
#' @method print learn_compare
#' @export

print.learn_compare <- function(x, ...) {

.print_header("Comparison Results")

# ===============================================
# General information
# ===============================================

.print_block("Data & Setup", function() {

  # Algorithms
  cat(
    "Algorithms:       ",
    x$n_models,
    "\n",
    sep = ""
  )

  # Models
  cat(
    "Models:           ",
    paste(x$algorithms, collapse = ", "),
    "\n",
    sep = ""
  )

  # Training samples
  if (length(unique(x$n_train)) == 1) {

    cat(
      "Training samples: ",
      x$n_train[1],
      "\n",
      sep = ""
    )

  } else {

    cat("Training samples:\n")

    for (i in seq_along(x$n_train)) {
      cat(
        "  ", names(x$n_train)[i], ": ",
        x$n_train[i], "\n",
        sep = ""
      )
    }
  }

  # Test samples
  if (length(unique(x$n_test)) == 1) {

    cat(
      "Test samples:     ",
      x$n_test[1],
      "\n",
      sep = ""
    )

  } else {

    cat("Test samples:\n")

    for (i in seq_along(x$n_test)) {
      cat(
        "  ", names(x$n_test)[i], ": ",
        x$n_test[i], "\n",
        sep = ""
      )
    }
  }

  # Classes
  cat(
    "Classes:          ",
    paste(x$classes, collapse = ", "),
    "\n",
    sep = ""
  )

  # Average metric
  cat(
    "Average metric:   ",
    x$average,
    "\n",
    sep = ""
  )

  # Positive class
  if (!is.null(x$positive_class)) {

    cat(
      "Positive class:   ",
      x$positive_class,
      "\n",
      sep = ""
    )
  }
})

# ===============================================
# Performance metrics
# ===============================================

.print_block("Performance Metrics", function() {

  if (
    is.list(x$metrics) &&
    !is.null(x$metrics$summary)
  ) {

    print(
      x$metrics$summary,
      row.names = FALSE
    )

  } else {

    print(x$metrics)

  }
})

# ===============================================
# ROC information
# ===============================================

.print_block("ROC Analysis", function() {

  if (is.null(x$roc)) {

    cat("ROC analysis was not performed.\n")

  } else {

    print(
      x$roc$summary,
      row.names = FALSE
    )

  }
})

# ===============================================
# Compatibility report
# ===============================================

.print_block("Compatibility", function() {

  checks <- x$compatibility$checks

  if (all(checks$status == "compatible")) {

    cat("All compatibility checks passed.\n")

  } else {

    print(
      checks[
        checks$status != "compatible",
        c("check", "status", "message"),
        drop = FALSE
      ],
      row.names = FALSE
    )
  }
})

invisible(x)

}
