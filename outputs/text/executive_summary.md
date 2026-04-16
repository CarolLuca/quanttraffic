# Executive Summary Snippets

- The project spans 6985228 accidents from 2016-01-14 to 2023-03-31.
- The top five states by recorded accidents are: CA, FL, TX, SC, NY.
- The most common weather bucket is `Clear/Cloudy`.
- The best severity model in this run is `Random Forest` with PR-AUC 0.688 on a holdout severe-share prevalence of 8.3%.
- The best national daily forecasting model in this run is `Random Forest Regressor` with RMSE 859.32.
- Robustness caveat: 53 sparse-coverage national days were excluded from daily signal, forecasting, and risk-day sections.
- The strongest discussion-ready unusual signal is `long_weekend` at 10.67% lift with 100% yearly sign stability.
- External mobility / energy / auto-market features were more useful for context than for monthly holdout RMSE improvement in this run.