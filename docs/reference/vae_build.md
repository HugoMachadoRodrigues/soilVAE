# Build a supervised VAE regression model (VAEReg)

Build a supervised VAE regression model (VAEReg)

## Usage

``` r
vae_build(
  input_dim,
  hidden_enc = c(512L, 256L),
  hidden_dec = c(256L, 512L),
  latent_dim = 32L,
  dropout = 0.1,
  lr = 0.001,
  beta_kl = 1,
  alpha_y = 1
)
```

## Arguments

- input_dim:

  Integer. Number of predictors (columns in X).

- hidden_enc:

  Integer vector. Encoder hidden layer sizes.

- hidden_dec:

  Integer vector. Decoder hidden layer sizes.

- latent_dim:

  Integer. Latent space dimension.

- dropout:

  Numeric. Dropout rate.

- lr:

  Numeric. Learning rate for Adam.

- beta_kl:

  Numeric. Weight for KL term.

- alpha_y:

  Numeric. Weight for supervised regression head loss.

## Value

A Python Keras model object (class `VAEReg`).
