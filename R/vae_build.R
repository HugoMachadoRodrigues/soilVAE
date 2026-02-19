#' Build a supervised VAE regression model (VAEReg)
#'
#' @param input_dim Integer. Number of predictors (columns in X).
#' @param hidden_enc Integer vector. Encoder hidden layer sizes.
#' @param hidden_dec Integer vector. Decoder hidden layer sizes.
#' @param latent_dim Integer. Latent space dimension.
#' @param dropout Numeric. Dropout rate.
#' @param lr Numeric. Learning rate for Adam.
#' @param beta_kl Numeric. Weight for KL term.
#' @param alpha_y Numeric. Weight for supervised regression head loss.
#' @return A Python Keras model object (class \\code{VAEReg}).
#' @export
vae_build <- function(input_dim,
                      hidden_enc = c(512L, 256L),
                      hidden_dec = c(256L, 512L),
                      latent_dim = 32L,
                      dropout = 0.1,
                      lr = 1e-3,
                      beta_kl = 1.0,
                      alpha_y = 1.0) {

  .soilvae_assert_tf()
  vae_py_module()

  keras <- reticulate::import("keras", convert = FALSE)

  m <- reticulate::py_eval(sprintf(
    "VAEReg(input_dim=%d, hidden_enc=%s, hidden_dec=%s, latent_dim=%d, dropout=%s, beta_kl=%s, alpha_y=%s)",
    as.integer(input_dim),
    paste0("(", paste(as.integer(hidden_enc), collapse = ","), ")"),
    paste0("(", paste(as.integer(hidden_dec), collapse = ","), ")"),
    as.integer(latent_dim),
    format(as.numeric(dropout), scientific = FALSE),
    format(as.numeric(beta_kl), scientific = FALSE),
    format(as.numeric(alpha_y), scientific = FALSE)
  ), convert = FALSE)

  m$compile(optimizer = keras$optimizers$Adam(learning_rate = as.numeric(lr)))
  m
}
