## Crypto currency forecasting challenge.

Cryptocurrency is a complex problem domain for the ML community due to the extreme volatility of the assets, the non-stationary nature of the data, the market and meme manipulation, the correlation between assets and the very fast changing market conditions. Here I record my progress on the G-research forecasting [competition](https://www.kaggle.com/c/g-research-crypto-forecasting/overview). 

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

- no duplicates 

- historical data set is until timestamp<1623542400 

- need to complete time-series data (there are many gaps)

- what do we predict here?

    Log returns (linearly residualized for each asset - idea is to see how good is the foreast on each asset separately).

    In order to analyze price changes for an asset we can deal with the price difference. However, different assets exhibit different price scales, so that the their returns are not readily comparable. We can solve this problem by computing the percentage change in price instead, also known as the return. This return coincides with the percentage change in our invested capital.

    Returns are widely used in finance, however log returns are preferred for mathematical modelling of time series, as they are additive across time. Also, while regular returns cannot go below -100%, log returns are not bounded.

- stationarity
A stationary behaviour of a system or a process is characterized by non-changing statistical properties over time such as the mean, variance and autocorrelation. On the other hand, a non-stationary behaviour is characterized by a continuous change of statistical properties over time. Stationarity is important because many useful analytical tools and statistical tests and models rely on it.

- crypto asset returns are highly correlated, following to a large extend the overall crypto market. 

- data is highly non-stationary, these results might vary a lot for different periods. Use cross-validation to avoid overfitting.

- why evaluation metrics is correlation between prediction and actual?
"While mean squared error, R^2, explained variance, and correlation are all very closely related, correlation has the useful property that it tends to normalize leading-order volatility out of the covariance between target and prediction. In financial markets (especially crypto ones!), predicting volatility is a difficult (but interesting!) question in its own right. By using correlation as a metric we hope to remove some noise from the prediction problem and provide a more stable metric to evaluate against."

- Target computation is derived from 'Close' price. Can either forecast target directly or
forecast 'Close' price.

## Challenges and Tasks
- Finish [tutorial](https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition) - completed
- reproduce target column - completed, see playground.py
- evaluation metric of the competition- completed check hosts [notebook](https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/291845)
- high but variable correlation between the assets. challenge is to perform forecasts in a highly non-stationary environment. Results may differ a lot on different periods (use CV).
- create baseline models: Arima, Prophet, Linear regression, Multiregression, VAR, HTS.
- Linear model in sklearn pipeline
- Use statsmodels linear model to get the summary and determine significant regressors