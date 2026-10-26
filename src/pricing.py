"""European option pricing: Black-Scholes (analytic) and Monte Carlo."""
import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_price(S, K, T, r, sigma, kind="call"):
    """Black-Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, kind="call"):
    """Analytic Greeks. Vega per 1.00 change in vol, theta per year."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)
    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T)
    if kind == "call":
        delta = norm.cdf(d1)
        theta = -S * pdf * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -S * pdf * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def mc_price(S, K, T, r, sigma, n_paths, kind="call", antithetic=False, seed=None):
    """Monte Carlo price under geometric Brownian motion.

    Returns (price, standard_error). With antithetic=True each normal draw z
    is paired with -z, which reduces variance at no extra cost.
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        z = rng.standard_normal(n_paths // 2)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    disc = np.exp(-r * T) * payoff
    if antithetic:  # average each pair first so the samples are independent
        half = len(disc) // 2
        disc = 0.5 * (disc[:half] + disc[half:])
    return disc.mean(), disc.std(ddof=1) / np.sqrt(len(disc))

def mc_price_cv(S, K, T, r, sigma, n_paths, kind="call", seed=None) :
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST -K, 0) if kind == "call" else np.maximum (K - ST, 0)
    Y = np.exp(-r * T) * payoff
    X = np.exp(-r * T) * ST

    b = np.cov(Y, X, ddof=1)[0,1] / np.var(X, ddof=1)
    Y_cv = Y - b *(X - S)
    return Y_cv.mean(), Y_cv.std(ddof=1) / np.sqrt(n_paths), b
