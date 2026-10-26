import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pricing import bs_greeks, bs_price, mc_price, mc_price_cv

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def test_known_value():
    # Textbook value (Hull): ATM call, S=K=100, T=1, r=5%, vol=20% -> 10.4506
    assert abs(bs_price(S, K, T, r, sigma) - 10.4506) < 1e-3


def test_put_call_parity():
    c = bs_price(S, K, T, r, sigma, "call")
    p = bs_price(S, K, T, r, sigma, "put")
    assert abs(c - p - (S - K * np.exp(-r * T))) < 1e-10


def test_mc_close_to_bs():
    price, se = mc_price(S, K, T, r, sigma, 400_000, seed=0)
    assert abs(price - bs_price(S, K, T, r, sigma)) < 4 * se


def test_delta_matches_finite_difference():
    h = 1e-4
    fd = (bs_price(S + h, K, T, r, sigma) - bs_price(S - h, K, T, r, sigma)) / (2 * h)
    assert abs(bs_greeks(S, K, T, r, sigma)["delta"] - fd) < 1e-6

def test_control_variate_close_to_bs_and_reduces_error():
    price, se, _ = mc_price_cv(S, K, T, r, sigma, 200_000, seed=0)
    _, se_plain = mc_price(S, K, T, r, sigma, 200_000, seed=0)
    assert abs(price - bs_price(S, K, T, r, sigma)) < 4 * se
    assert se < se_plain 