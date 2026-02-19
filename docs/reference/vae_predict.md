# Predict using a fitted VAEReg model

Predict using a fitted VAEReg model

## Usage

``` r
vae_predict(model, X)
```

## Arguments

- model:

  A fitted model returned by
  [`vae_build()`](https://hugomachadorodrigues.github.io/soilVAE/reference/vae_build.md)
  and trained with
  [`vae_fit()`](https://hugomachadorodrigues.github.io/soilVAE/reference/vae_fit.md).

- X:

  Matrix-like predictors.

## Value

Numeric vector of predictions.
