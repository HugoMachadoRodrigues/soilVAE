#' Fit a VAEReg model
#'
#' @param model A model returned by \\code{vae_build()}.
#' @param X Matrix-like predictors.
#' @param y Numeric response vector.
#' @param X_val Optional validation predictors.
#' @param y_val Optional validation response.
#' @param epochs Integer.
#' @param batch_size Integer.
#' @param patience Integer. Early stopping patience (only used if validation data provided).
#' @param verbose Integer verbosity passed to Keras.
#' @return Invisibly the fitted model.
#' @export
vae_fit <- function(model, X, y,
                    X_val = NULL, y_val = NULL,
                    epochs = 80L, batch_size = 64L,
                    patience = 10L, verbose = 0L) {

  .soilvae_assert_tf()
  stopifnot(!is.null(model))

  X <- as.matrix(X)
  y <- as.numeric(y)

  has_val <- !is.null(X_val) && !is.null(y_val)
  if (has_val) {
    X_val <- as.matrix(X_val)
    y_val <- as.numeric(y_val)
  }

  keras <- reticulate::import("keras", convert = FALSE)

  callbacks <- NULL
  if (has_val) {
    callbacks <- list(
      keras$callbacks$EarlyStopping(
        monitor = "val_loss",
        patience = as.integer(patience),
        restore_best_weights = TRUE
      )
    )
  }

  if (has_val) {
    model$fit(
      x = X, y = matrix(y, ncol = 1),
      validation_data = reticulate::tuple(X_val, matrix(y_val, ncol = 1)),
      epochs = as.integer(epochs),
      batch_size = as.integer(batch_size),
      verbose = as.integer(verbose),
      callbacks = callbacks
    )
  } else {
    model$fit(
      x = X, y = matrix(y, ncol = 1),
      epochs = as.integer(epochs),
      batch_size = as.integer(batch_size),
      verbose = as.integer(verbose)
    )
  }

  invisible(model)
}
