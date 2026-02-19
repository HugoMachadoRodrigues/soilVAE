#' Soil spectroscopy example dataset used in the soilVAE vignettes
#'
#' A small soil spectroscopy dataset packaged with **soilVAE** for demonstrating
#' typical spectral preprocessing (reflectance \eqn{\rightarrow} absorbance,
#' resampling, SNV, smoothing) and for comparing a classic PLS baseline model
#' against supervised VAE regression via \pkg{soilVAE}.
#'
#' The object `datsoilspc` contains:
#'
#' - `spc`: a numeric matrix (or data.frame) of reflectance spectra, with rows as
#'   samples and columns as wavelengths (nm). Column names should be interpretable
#'   as numeric wavelengths.
#' - `TotalCarbon`: a numeric vector with the soil total carbon content for each sample.
#'
#' Depending on the original source, additional columns may be present
#' (e.g., sample identifiers or other soil properties).
#'
#' @docType data
#' @name datsoilspc
#'
#' @format A data.frame or list containing at minimum:
#' \describe{
#'   \item{spc}{Numeric matrix/data.frame of reflectance spectra (samples \eqn{\times} wavelengths).}
#'   \item{TotalCarbon}{Numeric vector of total carbon values.}
#' }
#'
#' @details
#' The dataset is intended for examples and unit-sized demonstrations.
#' It is not meant to be a comprehensive soil spectral library.
#'
#' @examples
#' data("datsoilspc", package = "soilVAE")
#' str(datsoilspc)
#'
#' # basic plot of reflectance spectra
#' spc <- as.matrix(datsoilspc$spc)
#' wav <- as.numeric(colnames(spc))
#' matplot(wav, t(spc), type = "l", lty = 1,
#'         xlab = "Wavelength / nm", ylab = "Reflectance")
#'
#' @keywords datasets
NULL
