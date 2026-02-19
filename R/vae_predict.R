#' Predict using a fitted VAEReg model
#'
#' @param model A fitted model returned by `vae_build()` and trained with
#'   `vae_fit()`.
#' @param X Matrix-like predictors.
#' @return Numeric vector of predictions.
#' @export
vae_predict <- function(model, X) {
  .soilvae_assert_tf()
  X <- as.matrix(X)
  z <- model$encode(X, training = FALSE)[[1]]
  as.numeric(model$y_head(z, training = FALSE)$numpy())
}

#' Extract latent embeddings (z) from a fitted VAEReg model
#'
#' @param model A fitted model returned by `vae_build()` and trained with
#'   `vae_fit()`.
#' @param X Matrix-like predictors.
#' @return A numeric matrix of embeddings.
#' @export
vae_encode <- function(model, X) {
  .soilvae_assert_tf()
  X <- as.matrix(X)
  z <- model$encode(X, training = FALSE)[[1]]
  z$numpy()
}
