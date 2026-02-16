#' soilVAE: Supervised VAE regression via reticulate
#'
#' @name soilVAE
#' @docType package
NULL

# ---- Exports (exact) ----

#' Configure Python / reticulate for soilVAE
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

#' Build a supervised VAE regression model (VAEReg)
#' @export
vae_build <- function(input_dim,
                      hidden_enc = c(512L, 256L),
                      hidden_dec = c(256L, 512L),
                      latent_dim = 32L,
                      dropout = 0.1,
                      lr = 1e-3,
                      beta_kl = 1.0,
                      alpha_y = 1.0) {

  path <- system.file("python", "vae.py", package = "soilVAE")
  if (path == "") stop("Could not find inst/python/vae.py inside the package.")
  reticulate::source_python(path)

  keras <- reticulate::import("keras", convert = FALSE)

  m <- reticulate::py_eval(sprintf(
    "VAEReg(input_dim=%d, hidden_enc=%s, hidden_dec=%s, latent_dim=%d, dropout=%s, beta_kl=%s, alpha_y=%s)",
    as.integer(input_dim),
    paste0("(", paste(as.integer(hidden_enc), collapse=","), ")"),
    paste0("(", paste(as.integer(hidden_dec), collapse=","), ")"),
    as.integer(latent_dim),
    format(as.numeric(dropout), scientific = FALSE),
    format(as.numeric(beta_kl), scientific = FALSE),
    format(as.numeric(alpha_y), scientific = FALSE)
  ), convert = FALSE)

  m$compile(optimizer = keras$optimizers$Adam(learning_rate = as.numeric(lr)))
  m
}

#' Fit VAEReg
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
}

#' Predict y using VAEReg (via latent z -> y_head)
#' @export
vae_predict <- function(model, X) {
  X <- as.matrix(X)
  z <- model$encode(X, training = FALSE)[[1]]
  as.numeric(model$y_head(z, training = FALSE)$numpy())
}

#' Extract latent embeddings (z) from VAEReg
#' @export
vae_encode <- function(model, X) {
  X <- as.matrix(X)
  z <- model$encode(X, training = FALSE)[[1]]
  z$numpy()
}

#' Tune VAEReg on a Train/Validation split
#'
#' @param X_tr matrix
#' @param y_tr numeric vector
#' @param X_va matrix
#' @param y_va numeric vector
#' @param seed integer
#' @param grid_vae data.frame with required columns:
#'   hidden_enc (list), hidden_dec (list), latent_dim, dropout, lr, beta_kl, alpha_y, epochs, batch_size, patience
#' @return list(grid=grid_vae, tuning_df=results)
#' @export
tune_vae_train_val <- function(X_tr, y_tr, X_va, y_va, seed = 123, grid_vae) {

  set.seed(seed)

  X_tr <- as.matrix(X_tr)
  X_va <- as.matrix(X_va)
  y_tr <- as.numeric(y_tr)
  y_va <- as.numeric(y_va)

  # minimal dependency: use base data.frame
  grid <- as.data.frame(grid_vae, stringsAsFactors = FALSE)
  ncfg <- nrow(grid)
  rows <- vector("list", ncfg)

  for (i in seq_len(ncfg)) {
    g <- grid[i, , drop = FALSE]

    m <- vae_build(
      input_dim  = ncol(X_tr),
      hidden_enc = g$hidden_enc[[1]],
      hidden_dec = g$hidden_dec[[1]],
      latent_dim = as.integer(g$latent_dim),
      dropout    = as.numeric(g$dropout),
      lr         = as.numeric(g$lr),
      beta_kl    = as.numeric(g$beta_kl),
      alpha_y    = as.numeric(g$alpha_y)
    )

    vae_fit(
      model = m,
      X = X_tr, y = y_tr,
      X_val = X_va, y_val = y_va,
      epochs = as.integer(g$epochs),
      batch_size = as.integer(g$batch_size),
      patience = as.integer(g$patience),
      verbose = 0L
    )

    yhat <- vae_predict(m, X_va)

    # metrics (base)
    ok <- is.finite(y_va) & is.finite(yhat)
    yy <- y_va[ok]; pp <- yhat[ok]

    mse <- mean((yy - pp)^2)
    rmse <- sqrt(mse)
    ss_res <- sum((yy - pp)^2)
    ss_tot <- sum((yy - mean(yy))^2)
    r2 <- ifelse(ss_tot == 0, NA_real_, 1 - ss_res / ss_tot)
    rpiq <- stats::IQR(yy, na.rm = TRUE) / rmse

    rows[[i]] <- data.frame(
      cfg_id = i,
      RMSE_val = rmse,
      MSE_val  = mse,
      R2_val   = r2,
      RPIQ_val = rpiq,
      latent_dim = as.integer(g$latent_dim),
      dropout = as.numeric(g$dropout),
      lr      = as.numeric(g$lr),
      beta_kl = as.numeric(g$beta_kl),
      alpha_y = as.numeric(g$alpha_y),
      epochs = as.integer(g$epochs),
      batch_size = as.integer(g$batch_size),
      patience = as.integer(g$patience),
      hidden_enc_str = paste(g$hidden_enc[[1]], collapse = "-"),
      hidden_dec_str = paste(g$hidden_dec[[1]], collapse = "-"),
      stringsAsFactors = FALSE
    )
  }

  tuning_df <- do.call(rbind, rows)
  list(grid = grid_vae, tuning_df = tuning_df)
}

#' Select the best configuration from a tuning table
#'
#' @param tuning_df data.frame containing RMSE_val, R2_val, RPIQ_val
#' @param selection_metric one of: "euclid", "rmse", "r2", "rpiq"
#' @return list(best=one-row data.frame, best_score=numeric)
#' @export
select_best_from_grid <- function(tuning_df, selection_metric = c("euclid","rmse","r2","rpiq")) {

  selection_metric <- match.arg(selection_metric)
  df <- as.data.frame(tuning_df)

  req <- c("RMSE_val","R2_val","RPIQ_val")
  miss <- setdiff(req, names(df))
  if (length(miss) > 0) stop("select_best_from_grid(): missing columns: ", paste(miss, collapse = ", "))

  if (selection_metric == "rmse") {
    best_i <- which.min(df$RMSE_val)
    best_score <- df$RMSE_val[best_i]

  } else if (selection_metric == "r2") {
    best_i <- which.max(df$R2_val)
    best_score <- df$R2_val[best_i]

  } else if (selection_metric == "rpiq") {
    best_i <- which.max(df$RPIQ_val)
    best_score <- df$RPIQ_val[best_i]

  } else {
    rmse_min <- min(df$RMSE_val, na.rm = TRUE)
    rmse_max <- max(df$RMSE_val, na.rm = TRUE)
    rmse_n <- (df$RMSE_val - rmse_min) / ((rmse_max - rmse_min) + 1e-12)

    r2_n <- pmin(pmax(df$R2_val, 0), 1)

    rpiq_min <- min(df$RPIQ_val, na.rm = TRUE)
    rpiq_max <- max(df$RPIQ_val, na.rm = TRUE)
    rpiq_n <- (df$RPIQ_val - rpiq_min) / ((rpiq_max - rpiq_min) + 1e-12)

    eu <- sqrt((rmse_n - 0)^2 + (r2_n - 1)^2 + (rpiq_n - 1)^2)
    df$EUCLID_val <- eu

    best_i <- which.min(df$EUCLID_val)
    best_score <- df$EUCLID_val[best_i]
  }

  list(best = df[best_i, , drop = FALSE], best_score = best_score)
}
