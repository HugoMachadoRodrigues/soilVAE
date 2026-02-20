# Build a supervised VAE regression model (VAEReg)

Build a supervised VAE regression model (VAEReg)

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

  integer

- hidden_enc:

  integer vector

- hidden_dec:

  integer vector

- latent_dim:

  integer

- dropout:

  numeric

- lr:

  numeric learning rate

- beta_kl:

  numeric

- alpha_y:

  numeric

## Value

Python keras model object (VAEReg)
