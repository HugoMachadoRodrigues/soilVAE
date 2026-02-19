# Select the best configuration from a tuning table

Select the best configuration from a tuning table

Select the best configuration from a tuning table

## Usage

``` r
select_best_from_grid(
  tuning_df,
  selection_metric = c("euclid", "rmse", "r2", "rpiq")
)

select_best_from_grid(
  tuning_df,
  selection_metric = c("euclid", "rmse", "r2", "rpiq")
)
```

## Arguments

- tuning_df:

  Data frame containing `RMSE_val`, `R2_val`, and `RPIQ_val`.

- selection_metric:

  One of: `"euclid"`, `"rmse"`, `"r2"`, `"rpiq"`.

## Value

List with `best` (one-row data frame) and `best_score`.

List with `best` (one-row data frame) and `best_score`.
