#' Fit VAEReg
#'
#' @param model Python VAEReg object from vae_build()
#' @param X matrix (n x p)
#' @param y numeric vector (n)
#' @param X_val optional matrix
#' @param y_val optional numeric vector
#' @param epochs integer
#' @param batch_size integer
#' @param patience integer for early stopping (only if validation provided)
#' @param verbose 0/1/2
#' @return training history (Python object)
#' @export
vae_fit <- function(model, X, y,
                    X_val = NULL, y_val = NULL,
                    epochs = 80L, batch_size = 64L,
                    patience = 10L, verbose = 0L) {

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
