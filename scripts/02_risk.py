
from pathlib import Path
from scipy.stats import chi2, norm

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.stats import norm  
import pandas as pd


FIG = Path(__file__).resolve().parents[1] / "figures"

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
WEIGHTS = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
ALPHA = 0.99                 
PORTFOLIO_VALUE = 1_000_000  


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


cutoff = portfolio.quantile(1 - ALPHA)
var_hist = -cutoff                               
es_hist = -portfolio[portfolio <= cutoff].mean()   
n_tail = (portfolio <= cutoff).sum()

print(f"\nHistorical 1-day {ALPHA:.0%} VaR: {var_hist:.2%}  "
      f"= ${var_hist * PORTFOLIO_VALUE:,.0f}")
print(f"Historical 1-day {ALPHA:.0%} ES:  {es_hist:.2%}  "
      f"= ${es_hist * PORTFOLIO_VALUE:,.0f}")
print(f"Days in the tail: {n_tail}")


mu = portfolio.mean()
sd = portfolio.std()
z = norm.ppf(1 - ALPHA)                              
var_norm = -(mu + z * sd)
es_norm = -(mu - sd * norm.pdf(z) / (1 - ALPHA))

print(f"\nParametric 1-day {ALPHA:.0%} VaR: {var_norm:.2%}  "
      f"= ${var_norm * PORTFOLIO_VALUE:,.0f}")
print(f"Parametric 1-day {ALPHA:.0%} ES:  {es_norm:.2%}  "
      f"= ${es_norm * PORTFOLIO_VALUE:,.0f}")


breaches = (portfolio < -var_norm).sum()
expected = (1 - ALPHA) * len(portfolio)
print(f"Days worse than normal VaR: {breaches} (expected: {expected:.0f})")
print(f"Excess kurtosis: {portfolio.kurt():.1f} (normal distribution: 0)")


fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(portfolio, bins=150, density=True, alpha=0.6, label="Actual returns")
x = np.linspace(portfolio.min(), portfolio.max(), 500)
ax.plot(x, norm.pdf(x, mu, sd), color="black", label="Normal distribution")
ax.axvline(-var_hist, color="red", ls="--", label=f"Historical VaR = {var_hist:.2%}")
ax.axvline(-var_norm, color="orange", ls="--", label=f"Normal VaR = {var_norm:.2%}")
ax.set(xlabel="Daily portfolio return", ylabel="Density",
       title="Daily returns of an equal-weight portfolio, 2015-2025")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "var_histogram.png", dpi=150)
print("Figure saved to", FIG / "var_histogram.png")


WINDOW = 250
rolling = portfolio.rolling(WINDOW)
backtest = pd.DataFrame({
    "return": portfolio,
    "var_hist": -rolling.quantile(1 - ALPHA).shift(1),
    "var_norm": -(rolling.mean() + z * rolling.std()).shift(1),
}).dropna()

def kupiec_test(n_exceptions, n_days, p):
    """Kupiec proportion-of-failures test. Returns (LR statistic, p-value)."""
    x, T = n_exceptions, n_days
    p_hat = x / T
    log_l0 = (T - x) * np.log(1 - p) + x * np.log(p)
    log_l1 = (T - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    lr = -2 * (log_l0 - log_l1)
    return lr, 1 - chi2.cdf(lr, df=1)


n_days = len(backtest)
print(f"\nBacktest over {n_days} days, expected exceptions: {(1 - ALPHA) * n_days:.0f}")
for name in ["var_hist", "var_norm"]:
    exceptions = backtest["return"] < -backtest[name]
    n_exc = exceptions.sum()
    lr, p_value = kupiec_test(n_exc, n_days, 1 - ALPHA)
    verdict = "rejected" if p_value < 0.05 else "not rejected"
    print(f"{name}: {n_exc} exceptions ({n_exc / n_days:.2%}), "
          f"Kupiec LR = {lr:.2f}, p-value = {p_value:.3f} -> model {verdict}")

exc = backtest["return"] < -backtest["var_hist"]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(backtest.index, backtest["return"], lw=0.5, color="grey", label="Daily return")
ax.plot(backtest.index, -backtest["var_hist"], color="red", label="Historical VaR (rolling 1y)")
ax.plot(backtest.index, -backtest["var_norm"], color="orange", label="Normal VaR (rolling 1y)")
ax.scatter(backtest.index[exc], backtest["return"][exc], color="red", s=12, zorder=3,
           label="Exceptions")
ax.set(ylabel="Daily return", title="Backtest of 1-day 99% VaR")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "var_backtest.png", dpi=150)
print("Figure saved to", FIG / "var_backtest.png")