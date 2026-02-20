#' Predict y using VAEReg (via latent z -> y_head)
#'
#' @param model Python VAEReg
#' @param X matrix
#' @return numeric vector
#' @export
vae_predict <- function(model, X) {
  X <- as.matrix(X)
  z <- model$encode(X, training = FALSE)[[1]]
  as.numeric(model$y_head(z, training = FALSE)$numpy())
}

#' Extract latent embeddings (z) from VAEReg
#'
#' @param model Python VAEReg
#' @param X matrix
#' @return matrix
#' @export
vae_encode <- function(model, X) {
  X <- as.matrix(X)
  z <- model$encode(X, training = FALSE)[[1]]
  z$numpy()
}
