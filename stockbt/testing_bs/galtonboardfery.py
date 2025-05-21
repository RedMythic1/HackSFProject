import matplotlib.pyplot as plt
from math import comb
import numpy as np

def binomial_coefficients(m, p):
    coeffs = []
    for k in range(m + 1):
        coeff = comb(m, k) * (p ** k) * ((1 - p) ** (m - k))
        coeffs.append(coeff)
        
    return coeffs

if __name__ == "__main__":
    m = int(input("Enter number of steps (pegs): "))
    p = float(input("Enter probability to go right (as percent, e.g. 70 for 70%): ")) / 100
    coeffs = binomial_coefficients(m, p)
    x = np.arange(m + 1)
    plt.figure(figsize=(10, 5))
    plt.bar(x, coeffs, color='skyblue')
    plt.xlabel('Number of Rights (k)')
    plt.ylabel('Probability')
    plt.title(f'Galton Board Binomial Coefficients (steps={m}, p={p:.2f})')
    plt.tight_layout()
    plt.show()
