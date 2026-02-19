#' Configure Python / reticulate for soilVAE
#'
#' Selects a Python environment for use with \'reticulate\'.
#' This function **does not** install Python packages.
#'
#' @param python Path to a Python executable.
#' @param venv Name/path of a virtualenv.
#' @param conda Name of a conda environment.
#' @param persist If TRUE, stores the choice in \code{options()} for reuse within the current R session.
#' @return Invisibly TRUE.
#' @export
vae_configure <- function(python = NULL, venv = NULL, conda = NULL, persist = TRUE) {
  if (!is.null(python)) {
    reticulate::use_python(python, required = TRUE)
    if (isTRUE(persist)) {
      options(soilVAE.method = "python", soilVAE.python = python)
    }
    return(invisible(TRUE))
  }

  if (!is.null(venv)) {
    reticulate::use_virtualenv(venv, required = TRUE)
    if (isTRUE(persist)) {
      options(soilVAE.method = "venv", soilVAE.venv = venv)
    }
    return(invisible(TRUE))
  }

  if (!is.null(conda)) {
    reticulate::use_condaenv(conda, required = TRUE)
    if (isTRUE(persist)) {
      options(soilVAE.method = "conda", soilVAE.conda = conda)
    }
    return(invisible(TRUE))
  }

  # Reuse if previously set in this session.
  m <- getOption("soilVAE.method", NULL)
  if (identical(m, "python")) reticulate::use_python(getOption("soilVAE.python"), required = TRUE)
  if (identical(m, "venv"))   reticulate::use_virtualenv(getOption("soilVAE.venv"), required = TRUE)
  if (identical(m, "conda"))  reticulate::use_condaenv(getOption("soilVAE.conda"), required = TRUE)

  invisible(TRUE)
}

# ---- internal helpers --------------------------------------------------------

.soilvae_assert_py <- function() {
  if (!reticulate::py_available(initialize = FALSE)) {
    stop(
      "Python is not available to 'reticulate'.\n",
      "Configure it first, e.g.: soilVAE::vae_configure(conda = 'soilvae-tf')\n",
      "Then restart R and try again.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

.soilvae_assert_tf <- function() {
  .soilvae_assert_py()
  ok <- isTRUE(reticulate::py_module_available("tensorflow")) &&
        isTRUE(reticulate::py_module_available("keras"))
  if (!ok) {
    stop(
      "Python is available, but required modules were not found: 'tensorflow' and/or 'keras'.\n",
      "Install them in the selected environment and try again.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

vae_py_module <- function() {
  # Load vae.py shipped in inst/python/vae.py
  path <- system.file("python", "vae.py", package = "soilVAE")
  if (path == "") stop("Could not find inst/python/vae.py inside the package.", call. = FALSE)
  reticulate::source_python(path)
  invisible(TRUE)
}
