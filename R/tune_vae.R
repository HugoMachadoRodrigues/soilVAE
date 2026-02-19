#' Tune VAEReg on a train/validation split
#'
#' @param X_tr Train predictors matrix.
#' @param y_tr Train response numeric.
#' @param X_va Validation predictors matrix.
#' @param y_va Validation response numeric.
#' @param seed Integer seed.
#' @param grid_vae Data frame with required columns: \\code{hidden_enc} (list), \\code{hidden_dec} (list),
#'   \\code{latent_dim}, \\code{dropout}, \\code{lr}, \\code{beta_kl}, \\code{alpha_y}, \\code{epochs}, \\code{batch_size}, \\code{patience}.
#' @return A list with \\code{grid} (input grid) and \\code{tuning_df} (metrics per config).
#' @export
tune_vae_train_val <- function(X_tr, y_tr, X_va, y_va, seed = 123, grid_vae) {

  .soilvae_assert_tf()
  set.seed(seed)

  X_tr <- as.matrix(X_tr)
  X_va <- as.matrix(X_va)
  y_tr <- as.numeric(y_tr)
  y_va <- as.numeric(y_va)

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
#' @param tuning_df Data frame containing \\code{RMSE_val}, \\code{R2_val}, and \\code{RPIQ_val}.
#' @param selection_metric One of: \\code{"euclid"}, \\code{"rmse"}, \\code{"r2"}, \\code{"rpiq"}.
#' @return List with \\code{best} (one-row data frame) and \\code{best_score}.
#' @export
select_best_from_grid <- function(tuning_df, selection_metric = c("euclid", "rmse", "r2", "rpiq")) {

  selection_metric <- match.arg(selection_metric)
  df <- as.data.frame(tuning_df)

  req <- c("RMSE_val", "R2_val", "RPIQ_val")
  miss <- setdiff(req, names(df))
  if (length(miss) > 0) {
    stop("select_best_from_grid(): missing columns: ", paste(miss, collapse = ", "), call. = FALSE)
  }

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

    df$EUCLID_val <- sqrt((rmse_n - 0)^2 + (r2_n - 1)^2 + (rpiq_n - 1)^2)

    best_i <- which.min(df$EUCLID_val)
    best_score <- df$EUCLID_val[best_i]
  }

  list(best = df[best_i, , drop = FALSE], best_score = best_score)
}
