# Extract latent embeddings (z) from a fitted VAEReg model

Extract latent embeddings (z) from a fitted VAEReg model

## Usage

``` r
vae_encode(model, X)
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

A numeric matrix of embeddings.
