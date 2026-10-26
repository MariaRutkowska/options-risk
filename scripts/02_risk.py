import numpy as np
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
WEIGHTS = np.array([0.2, 0.2, 0.2, 0.2, 0.2])


prices = yf.download(TICKERS, start="2015-01-01", end="2025-12-31",
                     auto_adjust=True, threads=False)["Close"]
prices = prices[TICKERS]


missing = prices.columns[prices.isna().all()]
if len(missing) > 0:
    raise RuntimeError(f"Download failed for {list(missing)}. Run the script again.")
print(prices.tail())


returns = prices.pct_change(fill_method=None).dropna()

portfolio = returns @ WEIGHTS
print("Number of days:", len(portfolio))
print(f"Worst day: {portfolio.min():.2%} on {portfolio.idxmin().date()}")
print(f"Best day:  {portfolio.max():.2%} on {portfolio.idxmax().date()}")