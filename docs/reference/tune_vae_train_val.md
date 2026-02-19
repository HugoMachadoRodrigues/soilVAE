# Tune a supervised VAE on a fixed train/validation split

Fits the VAE under one or more hyperparameter configurations on a
**given** train/validation split and returns a tidy results table.

## Usage

``` r
tune_vae_train_val(X_tr, y_tr, X_va, y_va, seed = 123, grid_vae)

tune_vae_train_val(X_tr, y_tr, X_va, y_va, seed = 123, grid_vae)
```

## Arguments

- X_tr:

  Train predictors matrix.

- y_tr:

  Train response numeric.

- X_va:

  Validation predictors matrix.

- y_va:

  Validation response numeric.

- seed:

  Integer seed.

- grid_vae:

  Data frame with required columns: `hidden_enc` (list), `hidden_dec`
  (list), `latent_dim`, `dropout`, `lr`, `beta_kl`, `alpha_y`, `epochs`,
  `batch_size`, `patience`.

## Value

A list with:

- `grid`: the input grid

- `tuning_df`: metrics per configuration

A list with `grid` (input grid) and `tuning_df` (metrics per config).
