# =============================================================================
# LEARN DESIGN
# =============================================================================

#' Design and Diagnose a Classification Analysis
#'
#' Prepares predictor data, evaluates dataset characteristics, and
#' recommends an available classification function for downstream analysis.
#'
#' @description
#' \code{learn_design()} provides an exploratory design and diagnostic step
#' before model fitting. It does not train a predictive model. Instead, it
#' evaluates the structure of the supplied data and summarizes characteristics
#' that may influence the choice of a classification algorithm.
#'
#' The function expects a single data frame containing the target variable
#' and candidate predictors. Sample identifiers can optionally be supplied
#' through a dedicated column.
#'
#' @param data Data frame containing the target variable and candidate
#'   predictors. Each row represents a sample and each remaining column
#'   represents a candidate predictor.
#' @param target_var Character string specifying the column containing the
#'   classification target. The target must be a factor, character, or logical
#'   vector with at least two observed classes.
#' @param sample_col Optional character string specifying the column
#'   containing sample identifiers. When supplied, this column is used to
#'   assign row names and is removed from the predictor data. When
#'   \code{NULL}, existing row names are preserved; automatic row names are
#'   replaced with internal sample identifiers.
#' @param verbose Logical. Whether to display the diagnostic report after
#'   preparing the result (default: \code{TRUE}).
#'
#' @return
#' Invisibly returns an object of class \code{learn_design}, containing:
#' \describe{
#'   \item{predictors}{The data source used to construct the predictor matrix.}
#'   \item{target_var}{Name of the target variable.}
#'   \item{recommended_function}{A recommendation object containing the
#'   recommended learning function, alternative functions, and the rationale
#'   for the recommendation.}
#'   \item{diagnostics}{Dataset diagnostics, including missingness,
#'   zero-variance predictors, duplicated sample identifiers, dimensionality,
#'   predictor types, class distribution, and class balance.}
#' }
#'
#' @details
#' \strong{Classification only:}
#'
#' The current implementation supports classification tasks only.
#' The target variable must contain at least two observed classes.
#'
#' \strong{Sample identifiers:}
#'
#' When \code{sample_col} is supplied, its values must be non-missing,
#' non-empty, and unique. The identifier column is removed before diagnostic
#' evaluation. If \code{sample_col = NULL}, existing non-automatic row names
#' are preserved, while automatic row names are replaced with internal
#' sample identifiers.
#'
#' \strong{Diagnostics:}
#'
#' The function evaluates missing values, predictors containing missing
#' values, zero-variance predictors, sample identifier consistency,
#' dimensionality, predictor-type composition, and class distribution.
#' Missing values are reported but are not removed or imputed.
#'
#' \strong{Model recommendation:}
#'
#' Diagnostic information is passed to \code{.recommend_model()}, which
#' suggests one of the classification functions currently available in
#' \code{autolearnR}. Recommendations are heuristic and should be treated
#' as a design aid rather than a substitute for statistical judgment,
#' appropriate validation, or model comparison.
#'
#' @examples
#' \dontrun{
#' design <- learn_design(
#'   data = clinical_data,
#'   target_var = "Disease_Status",
#'   sample_col = "Sample"
#' )
#'
#' # Inspect the recommendation
#' design$recommended_function
#'
#' # Inspect dataset diagnostics
#' design$diagnostics
#' }
#'
#' @seealso
#' \code{\link{learn_compare}},
#' \code{\link{learn_tree}},
#' \code{\link{learn_logistic}},
#' \code{\link{learn_elastic_net}},
#' \code{\link{learn_random_forest}},
#' \code{\link{learn_svm}}
#'
#' @export

learn_design <- function(
    data,
    target_var,
    sample_col = NULL,
    verbose = TRUE
) {

# =============================================================================
# 1. Validate and prepare data
# =============================================================================

bundle <- .build_predictor_matrix(
  data = data,
  target_var = target_var,
  sample_col = sample_col
)

predictor_matrix <- bundle$predictor_matrix
y <- bundle$response

# =============================================================================
# 2. Validate classification target
# =============================================================================

if (
  !is.factor(y) &&
  !is.character(y) &&
  !is.logical(y)
) {
  stop(
    "'target_var' must be a factor, character, or logical vector."
  )
}

observed_classes <- unique(as.character(y[!is.na(y)]))

if (length(observed_classes) < 2) {
  stop(
    "Classification requires at least two observed classes."
  )
}

# =============================================================================
# 3. Predictor types
# =============================================================================

predictor_types <- data.frame(
  Predictor = names(predictor_matrix),
  Type = vapply(
    predictor_matrix,
    function(x) {
      if (is.numeric(x)) return("Numeric")
      if (is.factor(x)) return("Factor")
      if (is.character(x)) return("Character")
      if (is.logical(x)) return("Logical")
      class(x)[1]
    },
    character(1)
  ),
  stringsAsFactors = FALSE
)

# =============================================================================
# 4. Diagnostics
# =============================================================================

diagnostics <- .compute_diagnostics(
  predictor_matrix = predictor_matrix,
  y = y
)

diagnostics$predictor_types <- predictor_types

# =============================================================================
# 5. Model recommendation
# =============================================================================

recommendation <- .recommend_model(
  diagnostics = diagnostics
)

# =============================================================================
# 6. Return object
# =============================================================================

result <- list(
  predictors = bundle$predictor_sources,
  target_var = target_var,
  recommended_function = recommendation,
  diagnostics = diagnostics
)

class(result) <- "learn_design"

if (verbose) {
  print(result)
}

invisible(result)
}

# =============================================================================
# Print S3
# =============================================================================
#' Print method for learn_design objects
#'
#' Displays a formatted summary of dataset diagnostics and model
#' recommendations.
#'
#' @param x An object of class \code{learn_design}.
#' @param ... Further arguments (ignored).
#'
#' @return Invisibly returns \code{x}.
#' @method print learn_design
#' @export

print.learn_design <- function(x, ...) {

diagnostics <- x$diagnostics
recommendation <- x$recommended_function

.print_header("Learn Design Results")

# ===========================================
# Data & setup
# ===========================================

.print_block("Data & Setup", function() {

  cat("Target Variable:       ", x$target_var, "\n")
  cat("Samples:               ", diagnostics$n_samples, "\n")
  cat("Predictors:            ", diagnostics$n_predictors, "\n")
  cat("Numeric Predictors:    ", diagnostics$numeric_predictors, "\n")
  cat("Categorical Predictors:", diagnostics$categorical_predictors, "\n")
})

# ===========================================
# Dataset diagnostics
# ===========================================

.print_block("Dataset Diagnostics", function() {

  cat(
    "Missing Values:        ",
    sprintf("%.2f%%", 100 * diagnostics$overall_missing_fraction),
    "\n"
  )

  cat(
    "Predictors with NAs:   ",
    diagnostics$predictors_with_na,
    "\n"
  )

  cat(
    "Zero-Variance:         ",
    diagnostics$n_zero_variance,
    "\n"
  )

  cat(
    "Duplicated Samples:    ",
    diagnostics$duplicated_samples,
    "\n"
  )

  cat(
    "High Dimensionality:   ",
    ifelse(diagnostics$high_dimensional, "Yes", "No"),
    "\n"
  )

  cat(
    "Mixed Predictor Types: ",
    ifelse(diagnostics$mixed_predictors, "Yes", "No"),
    "\n"
  )

  cat(
    "High Correlation Pairs:",
    diagnostics$high_correlation_pairs,
    "\n"
  )
})

# ===========================================
# Correlation diagnostics
# ===========================================

if (diagnostics$high_correlation_pairs > 0) {

  .print_block("High Correlation Details", function() {

    cat(
      "Correlation threshold: |r| > ",
      diagnostics$correlation_threshold,
      "\n\n",
      sep = ""
    )

    print(
      diagnostics$high_correlation_details,
      row.names = FALSE
    )
  })
}

# ===========================================
# Classification summary
# ===========================================

.print_block("Classification Summary", function() {

  cat("Number of Classes:    ", diagnostics$n_classes, "\n")
  cat("Class Balance:        ", diagnostics$balance, "\n")

  cat("\nClass Distribution:\n")

  print(diagnostics$class_distribution)
})

# ===========================================
# Warnings
# ===========================================

if (length(diagnostics$warnings) > 0) {

  .print_block("Warnings", function() {

    for (warning in diagnostics$warnings) {
      cat("- ", warning, "\n", sep = "")
    }
  })
}

# ===========================================
# Model recommendation
# ===========================================

.print_block("Model Recommendation", function() {

  cat("Recommended Function:", recommendation$recommended_function, "\n")

  if (length(recommendation$alternatives) > 0) {

    cat("\nAlternative Functions:\n")

    for (alternative in recommendation$alternatives) {
      cat("- ", alternative, "\n", sep = "")
    }
  }

  cat("\nReason:\n")
  cat(recommendation$reason, "\n")
})

invisible(x)
}
