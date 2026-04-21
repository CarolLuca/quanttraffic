# Executive Summary Snippets

- The project spans 6985228 accidents from 2016-01-14 to 2023-03-31.
- The top five states by recorded accidents are: CA, FL, TX, SC, NY.
- The most common weather bucket is `Clear/Cloudy`.
- The best severity model in this run is `XGBoost (GPU)` with PR-AUC 0.683 on a holdout severe-share prevalence of 8.3%.
- The best national daily forecasting model in this run is `Poisson Regressor` with RMSE 1061.65.
- Robustness caveat: 53 sparse-coverage national days were excluded from daily signal, forecasting, and risk-day sections.
- The strongest discussion-ready unusual signal is `long_weekend` at 10.67% lift with 100% yearly sign stability.
- External mobility / energy / auto-market features improved the best monthly RMSE by 25.55% versus lag-only monthly features.