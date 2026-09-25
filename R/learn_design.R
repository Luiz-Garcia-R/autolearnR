#' Design and Diagnose a Machine Learning Analysis
#'
#' Prepares predictor data, identifies the learning objective, evaluates basic
#' dataset characteristics, and recommends an appropriate learning function
#' for downstream analysis.
#'
#' \code{learn.design()} is intended as an entry point to the learning
#' framework. It can integrate molecular, clinical, phenotypic, or other
#' structured predictors from feature and metadata tables. Rather than fitting
#' a predictive model directly, the function evaluates the structure of the
#' available data and summarizes the main characteristics relevant to model
#' selection.
#'
#' The function can use predictors derived from feature data, metadata, or
#' both. When both sources are supplied, samples are matched using
#' \code{sample_column}. The resulting predictor matrix is then evaluated for
#' missing values, zero-variance predictors, duplicated samples,
#' dimensionality, and predictor-type composition. The learning objective can
#' be supplied explicitly or inferred from the available outcome information.
#'
#' @param feature_data Optional data frame containing feature-level predictors,
#'   such as gene expression, protein abundance, metabolite measurements, or
#'   other quantitative variables. If \code{sample_column} is present, its
#'   values are used as sample identifiers and the column is removed from the
#'   predictor data.
#' @param metadata Optional data frame containing sample-level metadata,
#'   clinical variables, phenotypic variables, or other candidate predictors.
#'   The column specified by \code{sample_column} must be present when
#'   \code{metadata} is supplied.
#' @param predictors Character vector specifying which data sources should
#'   contribute predictors. Supported values are \code{"feature_data"} and
#'   \code{"metadata"}. Multiple sources can be selected simultaneously.
#' @param output Optional specification of the outcome variable used to define
#'   the learning task. The interpretation of this argument is delegated to
#'   \code{detect_task()}. It may represent a variable contained in the
#'   metadata or another object supported by the learning framework.
#' @param objective Optional character specifying the learning objective.
#'   Supported values are \code{"classification"}, \code{"regression"}, and
#'   \code{"clustering"}. If \code{NULL}, the objective is inferred from the
#'   supplied data and outcome.
#' @param sample_column Character. Name of the column containing sample
#'   identifiers (default: \code{"Sample"}). Sample identifiers are used to
#'   match feature data and metadata and are removed from the predictor matrix
#'   when appropriate.
#' @param verbose Logical. Whether to print the design report and diagnostic
#'   information to the console (default: \code{TRUE}).
#'
#' @return
#' Invisibly returns an object of class \code{learn_design}.
#'
#' The \code{learn_design} object contains:
#' \describe{
#'   \item{task}{Detected or explicitly specified learning objective:
#'   classification, regression, or clustering.}
#'   \item{predictors}{Information describing the data sources used to construct
#'   the predictor matrix.}
#'   \item{output}{The outcome specification supplied to the function.}
#'   \item{recommended_function}{A recommendation object returned by
#'   \code{recommend_model()}, including the recommended learning function,
#'   alternative functions, and the rationale for the recommendation.}
#'   \item{diagnostics}{Diagnostic information describing the predictor matrix
#'   and available outcome, including missingness, zero-variance predictors,
#'   duplicated samples, dimensionality, predictor types, and classification
#'   balance when applicable.}
#' }
#'
#' @details
#' \strong{Role in the learning framework:}
#'
#' \code{learn.design()} is intended to serve as a design and diagnostic layer
#' before model fitting. It does not train a predictive model itself. Instead,
#' it organizes the available data, identifies the learning problem, evaluates
#' relevant dataset properties, and suggests a downstream learning function.
#'
#' \strong{Predictor sources:}
#'
#' Predictors can originate from \code{feature_data}, \code{metadata}, or both.
#' This allows molecular measurements and sample-level variables to be handled
#' within the same framework. When both sources are supplied, only samples
#' present in both tables are retained.
#'
#' \strong{Sample matching:}
#'
#' Samples are matched using the column specified by \code{sample_column}.
#' When both feature data and metadata are supplied, their row names are used
#' as sample identifiers after preparation. Samples without a corresponding
#' entry in the other data source are excluded from the combined predictor
#' matrix.
#'
#' \strong{Learning objective:}
#'
#' The learning objective can be explicitly specified through
#' \code{objective}. If it is not supplied, the function uses
#' \code{detect_task()} to infer whether the analysis corresponds to
#' classification, regression, or clustering based on the available outcome
#' information.
#'
#' \strong{Diagnostics:}
#'
#' The function evaluates general properties of the predictor matrix,
#' including missing-value burden, predictors containing missing values,
#' zero-variance predictors, duplicated samples, dimensionality, and the
#' presence of mixed predictor types. For classification tasks, the outcome
#' balance is also summarized.
#'
#' \strong{Model recommendation:}
#'
#' The diagnostic information is passed to \code{recommend_model()}, which
#' proposes a downstream learning function based on the detected task and
#' dataset characteristics. The recommendation is intended as a design aid
#' rather than a substitute for statistical judgment or model validation.
#'
#' @examples
#' \dontrun{
#'
#' # Example 1: Design a classification analysis using metadata
#' design <- learn.design(
#'   metadata = clinical_data,
#'   predictors = "metadata",
#'   output = "Disease_Status"
#' )
#'
#' # Example 2: Use molecular features and metadata together
#' design <- learn.design(
#'   feature_data = expression_data,
#'   metadata = clinical_data,
#'   predictors = c("feature_data", "metadata"),
#'   output = "Treatment_Response"
#' )
#'
#' # Example 3: Explicitly define a regression task
#' design <- learn.design(
#'   feature_data = expression_data,
#'   metadata = clinical_data,
#'   predictors = c("feature_data", "metadata"),
#'   output = "Antibody_Titer",
#'   objective = "regression"
#' )
#'
#' # Example 4: Design a clustering analysis without an outcome
#' design <- learn.design(
#'   feature_data = expression_data,
#'   predictors = "feature_data",
#'   objective = "clustering"
#' )
#'
#' # Inspect the recommended downstream function
#' design$recommended_function
#'
#' # Inspect dataset diagnostics
#' design$diagnostics
#' }
#'
#' @export

learn.design <- function(
    feature_data = NULL,
    metadata = NULL,
    predictors = c("feature_data", "metadata"),
    output = NULL,
    objective = NULL,
    sample_column = "Sample",
    verbose = TRUE
){

# =============================================================================
# 1. Argument validation
# =============================================================================

predictors <- match.arg(
  predictors,
  choices = c("feature_data", "metadata"),
  several.ok = TRUE
)

if (is.null(feature_data) && is.null(metadata)) {
  stop("At least one of 'feature_data' or 'metadata' must be supplied.")
}

if (!is.null(feature_data) && !is.data.frame(feature_data)) {
  stop("'feature_data' must be a data.frame.")
}

if (!is.null(metadata) && !is.data.frame(metadata)) {
  stop("'metadata' must be a data.frame.")
}

if (!is.null(objective)) {
  objective <- match.arg(
    objective,
    c("classification",
      "regression",
      "clustering")
  )
}

# =============================================================================
# 2. Prepare feature matrix
# =============================================================================

if (!is.null(feature_data)) {

  if (sample_column %in% colnames(feature_data)) {

    rownames(feature_data) <- feature_data[[sample_column]]

    feature_data[[sample_column]] <- NULL

  }

}

# =============================================================================
# 3. Prepare metadata
# =============================================================================

if (!is.null(metadata)) {

  if (!(sample_column %in% colnames(metadata))) {

    stop(
      paste0(
        "'",
        sample_column,
        "' was not found in metadata."
      )
    )

  }

  rownames(metadata) <- metadata[[sample_column]]

}

# =============================================================================
# 4. Check sample matching
# =============================================================================

if (!is.null(feature_data) &&
    !is.null(metadata)) {

  common_samples <-
    intersect(
      rownames(feature_data),
      rownames(metadata)
    )

  if (length(common_samples) == 0) {

    stop(
      "No matching samples were found between feature_data and metadata."
    )

  }

  feature_data <-
    feature_data[
      common_samples,
      ,
      drop = FALSE
    ]

  metadata <-
    metadata[
      common_samples,
      ,
      drop = FALSE
    ]

}

# =============================================================================
# 5. Build predictor matrix
# =============================================================================

bundle <- build_predictor_matrix(
  feature_data = feature_data,
  metadata = metadata,
  predictors = predictors,
  output = output,
  sample_column = sample_column
)

predictor_matrix <- bundle$predictor_matrix
predictor_sources <- bundle$predictor_sources

# =============================================================================
# 6. Determine learning objective (refactored)
# =============================================================================

task <- detect_task(
  metadata = metadata,
  output = output,
  objective = objective
)

# =============================================================================
# 7. Basic dataset information
# =============================================================================

n_samples <- nrow(predictor_matrix)
n_predictors <- ncol(predictor_matrix)

predictor_types <- data.frame(
  Predictor = names(predictor_matrix),
  Type = sapply(predictor_matrix, function(x){

    if (is.numeric(x)) return("Numeric")
    if (is.factor(x)) return("Factor")
    if (is.character(x)) return("Character")

    class(x)[1]
  }),
  stringsAsFactors = FALSE
)

numeric_predictors <- sum(predictor_types$Type == "Numeric")
categorical_predictors <- n_predictors - numeric_predictors

# =============================================================================
# 8. Initial report
# =============================================================================

if(verbose){
  message("\n")
  message("==========================================")
  message("          Learn Design Report")
  message("==========================================")
  message("Learning task : ", task)
  message("Samples       : ", n_samples)
  message("Predictors    : ", n_predictors)
  message("Numeric       : ", numeric_predictors)
  message("Categorical   : ", categorical_predictors)
  message("")
  message("Predictors from : ", paste(predictor_sources, collapse = " + "))
  message("")
}

# =============================================================================
# 9. Diagnostics
# =============================================================================

diagnostics <- compute_diagnostics(
  predictor_matrix = predictor_matrix,
  metadata = metadata,
  output = output
)

# =============================================================================
# 10. Model recommendation
# =============================================================================

recommendation <- recommend_model(
  diagnostics = diagnostics,
  task = task
)

# =============================================================================
# 11. Console report
# =============================================================================

if (verbose) {

  message("")
  message("==========================================")
  message("              DATA DIAGNOSTIC")
  message("==========================================")

  message("Task   : ", task)
  message("Samples: ", n_samples)
  message("Predictors: ", n_predictors)

  message("")

# ===============================================
# 11.1. Diagnostics
# ===============================================

message("Diagnostics")
message("------------------------------------------")

message("Missing values        : ", diagnostics$overall_missing_fraction)
message("Predictors with NAs   : ", diagnostics$predictors_with_na)
message("Zero variance         : ", diagnostics$n_zero_variance)
message("Duplicated samples    : ", diagnostics$duplicated_samples)
message("High dimensionality   : ",
        ifelse(diagnostics$high_dimensional, "Yes", "No"))

message("Mixed predictor types : ",
        ifelse(diagnostics$mixed_predictors, "Yes", "No"))

message("")

  # ------------------------------------
  # 11.1.1. Outcome summary
  # ------------------------------------

  if(task == "classification"){

    message("Classification summary")
    message("------------------------------------------")

    message("Balance : ", diagnostics$balance)

    message("")

  }

  # ------------------------------------
  # 11.1.2. Warnings
  # ------------------------------------

  warnings <- diagnostics$warnings

  if (length(warnings) > 0) {
    warnings <- as.character(warnings)

    message("Warnings")
    message("------------------------------------------")

    for (i in diagnostics$warnings) {
      message(i)
    }
  }

  # ------------------------------------
  # 11.1.3. recommended_function
  # ------------------------------------

  message("Recommended function")
  message("------------------------------------------")

  message(recommendation$recommended_function)

  message("")
  message("Reason")
  message(recommendation$reason)

  message("")
  message("Alternative functions")

  for(i in recommendation$alternatives){

    message(i)

  }

  message("")
  message("Suggested workflow")
  message("learn.design()")
  message(recommendation$recommended_function, "()")
  message("==========================================")
  message("")

}

# =============================================================================
# 12. Return object
# =============================================================================

result <- list(

  task = task,
  predictors = predictor_sources,
  output = output,

  recommended_function = recommendation,

  diagnostics = diagnostics

)

class(result) <- "learn_design"

invisible(result)

}
