#' Configure Python / reticulate for soilVAE
#'
#' @param python Path to python executable (optional).
#' @param venv Name of virtualenv to use (optional).
#' @param conda Name of conda env to use (optional).
#' @return TRUE invisibly.
#' @export
vae_configure <- function(python = NULL, venv = NULL, conda = NULL) {
  if (!is.null(python)) {
    reticulate::use_python(python, required = TRUE)
  } else if (!is.null(venv)) {
    reticulate::use_virtualenv(venv, required = TRUE)
  } else if (!is.null(conda)) {
    reticulate::use_condaenv(conda, required = TRUE)
  }
  invisible(TRUE)
}

# internal: load vae.py from inst/python
vae_py_module <- function() {
  path <- system.file("python", "vae.py", package = "soilVAE")
  if (path == "") stop("Could not find inst/python/vae.py inside the package.")
  reticulate::source_python(path)
  invisible(TRUE)
}
