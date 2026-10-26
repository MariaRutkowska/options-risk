
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pricing import bs_greeks, bs_price, mc_price, mc_price_cv

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

exact = bs_price(S, K, T, r, sigma)
print(f"Black-Scholes call price: {exact:.4f}")


ns = np.logspace(2, 6, 13).astype(int)
plain = np.array([mc_price(S, K, T, r, sigma, n, seed=1) for n in ns])
anti = np.array([mc_price(S, K, T, r, sigma, n, antithetic=True, seed=1) for n in ns])
cv = np.array([mc_price_cv(S, K, T, r, sigma, n, seed=1)[:2] for n in ns])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.axhline(exact, color="k", lw=1, label="Black-Scholes")
ax1.errorbar(ns, plain[:, 0], yerr=1.96 * plain[:, 1], fmt="o-", ms=3, capsize=2, label="MC (95% CI)")
ax1.set(xscale="log", xlabel="Number of paths", ylabel="Call price", title="MC price converges to Black-Scholes")
ax1.legend()

ax2.loglog(ns, plain[:, 1], "o-", ms=3, label="Plain MC")
ax2.loglog(ns, anti[:, 1], "s-", ms=3, label="Antithetic variates")
ax2.loglog(ns, cv[:,1], "^-", ms=3, label="Control variate")
ax2.loglog(ns, plain[0, 1] * np.sqrt(ns[0] / ns), "k--", lw=1, label=r"$\propto 1/\sqrt{N}$")
ax2.set(xlabel="Number of paths", ylabel="Standard error", title="Error falls as 1/sqrt(N)")
ax2.legend()
fig.tight_layout()
fig.savefig(FIG / "mc_convergence.png", dpi=150)

reduction = (plain[-1, 1] / anti[-1, 1]) ** 2
print(f"MC at {ns[-1]:,} paths: {plain[-1, 0]:.4f} ± {plain[-1, 1]:.4f}")
print(f"Antithetic variates cut variance by a factor of {reduction:.1f}")
print(f"Control variate cut variance by a factor of {(plain[-1,1] / cv[-1,1]) ** 2:.1f}")


spots = np.linspace(60, 140, 200)
g = bs_greeks(spots, K, T, r, sigma)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, name in zip(axes, ["delta", "gamma", "vega"]):
    ax.plot(spots, g[name])
    ax.axvline(K, color="grey", ls=":", lw=1)
    ax.set(title=name.capitalize(), xlabel="Stock price S")
fig.suptitle("Greeks of a European call (K=100, T=1y, vol=20%)")
fig.tight_layout()
fig.savefig(FIG / "greeks.png", dpi=150)

print("Greeks at S=100:", {k: round(float(v), 4) for k, v in bs_greeks(S, K, T, r, sigma).items()})
print(f"Figures saved to {FIG}")

