# Fit VAEReg

Fit VAEReg

Fit VAEReg

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

  Python VAEReg object from vae_build()

- X:

  matrix (n x p)

- y:

  numeric vector (n)

- X_val:

  optional matrix

- y_val:

  optional numeric vector

- epochs:

  integer

- batch_size:

  integer

- patience:

  integer for early stopping (only if validation provided)

- verbose:

  0/1/2

## Value

training history (Python object)
