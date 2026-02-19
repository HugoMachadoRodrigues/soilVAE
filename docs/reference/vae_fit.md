# Fit a VAEReg model

Fit a VAEReg model

## Usage

``` r
vae_fit(
  model,
  X,
  y,
  X_val = NULL,
  y_val = NULL,
  epochs = 80L,
  batch_size = 64L,
  patience = 10L,
  verbose = 0L
)
```

## Arguments

- model:

  A model returned by
  [`vae_build()`](https://hugomachadorodrigues.github.io/soilVAE/reference/vae_build.md).

- X:

  Matrix-like predictors.

- y:

  Numeric response vector.

- X_val:

  Optional validation predictors.

- y_val:

  Optional validation response.

- epochs:

  Integer.

- batch_size:

  Integer.

- patience:

  Integer. Early stopping patience (only used if validation data
  provided).

- verbose:

  Integer verbosity passed to Keras.

## Value

Invisibly the fitted model.
