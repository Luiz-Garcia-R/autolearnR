.onAttach <- function(libname, pkgname) {

  version_text <- paste0(
    "autolearnR v",
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
  )
}
