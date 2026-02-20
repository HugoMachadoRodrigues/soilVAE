# Package index

## Core API

- [`vae_configure()`](https://hugomachadorodrigues.github.io/soilVAE/reference/vae_configure.md)
  : Configure Python / reticulate for soilVAE
- [`vae_build()`](https://hugomachadorodrigues.github.io/soilVAE/reference/vae_build.md)
  : Build a supervised VAE regression model (VAEReg)
- [`vae_fit()`](https://hugomachadorodrigues.github.io/soilVAE/reference/vae_fit.md)
  : Fit VAEReg
- [`vae_predict()`](https://hugomachadorodrigues.github.io/soilVAE/reference/vae_predict.md)
  : Predict y using VAEReg (via latent z -\> y_head)
- [`vae_encode()`](https://hugomachadorodrigues.github.io/soilVAE/reference/vae_encode.md)
  : Extract latent embeddings (z) from VAEReg

## Tuning

- [`tune_vae_train_val()`](https://hugomachadorodrigues.github.io/soilVAE/reference/tune_vae_train_val.md)
  : Tune VAEReg on a Train/Validation split
- [`select_best_from_grid()`](https://hugomachadorodrigues.github.io/soilVAE/reference/select_best_from_grid.md)
  : Select the best configuration from a tuning table

## Data

- [`datsoilspc`](https://hugomachadorodrigues.github.io/soilVAE/reference/datsoilspc.md)
  : Soil spectroscopy example dataset used in the soilVAE vignettes
