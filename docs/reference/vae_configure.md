# Configure Python / reticulate for soilVAE

Selects a Python environment for use with \\reticulate\\. This function
**does not** install Python packages.

## Usage

``` r
vae_configure(python = NULL, venv = NULL, conda = NULL, persist = TRUE)
```

## Arguments

- python:

  Path to a Python executable.

- venv:

  Name/path of a virtualenv.

- conda:

  Name of a conda environment.

- persist:

  If TRUE, stores the choice in
  [`options()`](https://rdrr.io/r/base/options.html) for reuse within
  the current R session.

## Value

Invisibly TRUE.
