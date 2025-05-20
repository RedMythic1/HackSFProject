import math
from scipy.stats import norm
import numpy as np

def binomial_coefficients(m, p):
    coeffs = []
    for k in range(m + 1):
        coeff = math.comb(m, k) * (p ** k) * ((1 - p) ** (m - k))
        coeffs.append(coeff)
    return coeffs

def gaussian_approximation(m, left_pct):
    q = left_pct / 100
    p = 1 - q
    mu = m * p
    sigma2 = m * p * q
    sigma = np.sqrt(sigma2)
    k = np.arange(m + 1)
    coeff = 1 / np.sqrt(2 * np.pi * sigma2)
    P = coeff * np.exp(-((k - mu) ** 2) / (2 * sigma2))
    return k, P

def bellcurve_allocation(dict):
    balance = dict["balance"]
    stocks = dict["stocks"]
    raw_allocs = []
    tickers = []
    m = 1000  # Number of steps for binomial/Gaussian
    for stock in stocks:
        ticker = stock["ticker"]
        risk = stock["risk"]
        left_pct = risk  # risk is percent chance to go down (left)
        q = left_pct / 100
        p = 1 - q
        mu = m * p
        sigma2 = m * p * q
        print(f"\nStock: {ticker}")
        print(f"Risk (left %): {left_pct}")
        print(f"Gaussian Approximation Equation:")
        print(f"P(k) ≈ 1 / sqrt(2π * {m} * {p:.4f} * {q:.4f}) * exp(- (k - {m} * {p:.4f})^2 / (2 * {m} * {p:.4f} * {q:.4f}))")
        # Calculate binomial and Gaussian (not used for allocation, just for info)
        binom = binomial_coefficients(m, p)
        k, gauss = gaussian_approximation(m, left_pct)
        # Raw allocation (can be replaced with your own logic)
        alloc = balance * (p ** 2)
        raw_allocs.append(alloc)
        tickers.append(ticker)
    # Quadratic scaling if needed
    sum_raw = sum(raw_allocs)
    if sum_raw > balance:
        squares = [a ** 2 for a in raw_allocs]
        sum_squares = sum(squares)
        scaled_allocs = [math.floor(balance * (s / sum_squares)) for s in squares]
        # Adjust last allocation to ensure total does not exceed balance
        diff = balance - sum(scaled_allocs)
        if diff > 0:
            scaled_allocs[-1] += diff
    else:
        scaled_allocs = [math.floor(a) for a in raw_allocs]
        diff = balance - sum(scaled_allocs)
        if diff > 0:
            scaled_allocs[-1] += diff
    return_dict = {tickers[i]: scaled_allocs[i] for i in range(len(tickers))}
    bomboclat = balance - sum(scaled_allocs)
    print(f"Remainder (bomboclat): {bomboclat}")
    return_dict["bomboclat"] = bomboclat
    return return_dict

def input_to_dict():
    balance = float(input("Enter the balance: "))
    num_stocks = int(input("Enter the number of stocks: "))
    stocks = []
    for i in range(num_stocks):
        ticker = input(f"Enter the ticker for stock {i+1}: ")
        risk = float(input(f"Enter the risk for stock {i+1}: "))
        stocks.append({"ticker": ticker, "risk": risk})
    return {"balance": balance, "stocks": stocks}

if __name__ == "__main__":
    dict = input_to_dict()
    return_dict = bellcurve_allocation(dict)
    print(return_dict)

