<<<<<<< HEAD
.onLoad <- function(libname, pkgname) {
    invisible(NULL)
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage(
    "\n",
    crayon::green("autolearnR "), "loaded successfully!\n",
    "--------------------------------------------------\n",
    "A package for streamlined machine learning data analysis.\n",
    "Use ", crayon::green("?autolearnR"), " for general help.\n",
    "--------------------------------------------------\n"
=======
.onAttach <- function(libname, pkgname) {

  version_text <- paste0(
    "autolearnR v.",
    utils::packageVersion("autolearnR")
  )

  if (requireNamespace("crayon", quietly = TRUE)) {
    version_text <- crayon::green(version_text)
  }

  packageStartupMessage(
    "\n",
    version_text, " loaded successfully!\n",
    "--------------------------------------------------\n",
    "A package for streamlined machine learn data analysis.\n",
    "GitHub: https://github.com/Luiz-Garcia-R/autolearnR\n"
>>>>>>> aabaaa7 (refactor: redesign 'learn.rf()' function)
  )
}
