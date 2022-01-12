## Crypto currency forecasting challenge.

Cryptocurrency is a complex problem domain for the ML community due to the extreme volatility of the assets, the non-stationary nature of the data, the market and meme manipulation, the correlation between assets and the very fast changing market conditions.

## Notes on the project.

- historical dataset train.csv

    timestamp: All timestamps are returned as second Unix timestamps (the number of seconds elapsed since 1970-01-01 00:00:00.000 UTC). Timestamps in this dataset are multiple of 60, indicating minute-by-minute data.
    Asset_ID: The asset ID corresponding to one of the crytocurrencies (e.g. Asset_ID = 1 for Bitcoin). The mapping from Asset_ID to crypto asset is contained in asset_details.csv.
    Count: Total number of trades in the time interval (last minute).
    Open: Opening price of the time interval (in USD).
    High: Highest price reached during time interval (in USD).
    Low: Lowest price reached during time interval (in USD).
    Close: Closing price of the time interval (in USD).
    Volume: Quantity of asset bought or sold, displayed in base currency USD.
    VWAP: The average price of the asset over the time interval, weighted by volume. VWAP is an aggregated form of trade data.
    Target: Residual log-returns for the asset over a 15 minute horizon.

- need to complete time-series data (there are many gaps)

- what do we predict here?

    Log returns

    In order to analyze price changes for an asset we can deal with the price difference. However, different assets exhibit different price scales, so that the their returns are not readily comparable. We can solve this problem by computing the percentage change in price instead, also known as the return. This return coincides with the percentage change in our invested capital.

    Returns are widely used in finance, however log returns are preferred for mathematical modelling of time series, as they are additive across time. Also, while regular returns cannot go below -100%, log returns are not bounded.

- stationarity
A stationary behaviour of a system or a process is characterized by non-changing statistical properties over time such as the mean, variance and autocorrelation. On the other hand, a non-stationary behaviour is characterized by a continuous change of statistical properties over time. Stationarity is important because many useful analytical tools and statistical tests and models rely on it.

- crypto asset returns are highly correlated, following to a large extend the overall crypto market. 

- data is highly non-stationary, these results might vary a lot for different periods. Use cross-validation to avoid overfitting.
## Challenges

- high but variable correlation between the assets. Q: how to perform forecasts in a highly non-stationary environment?
