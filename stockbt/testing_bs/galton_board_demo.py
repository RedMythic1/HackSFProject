import matplotlib.pyplot as plt
from math import comb
import numpy as np

def galton_board_distribution_risk(steps, risk_percent):
    """
    Simulate a Galton board with a given number of steps and risk_percent chance to go down at each step.
    Returns a list of probabilities for each possible outcome (number of ups from 0 to steps).
    """
    p_down = risk_percent / 100
    p_up = 1 - p_down
    distribution = []
    for k in range(steps + 1):
        # k = number of ups, (steps-k) = number of downs
        prob = comb(steps, k) * (p_up ** k) * (p_down ** (steps - k))
        distribution.append(prob)
    return distribution

if __name__ == "__main__":
    print("--- Galton Board Distribution Overlay (Risk 1-100%) ---")
    steps = int(input("Enter number of steps for Galton board: "))
    x = np.arange(steps + 1)
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('viridis')
    for risk in range(1, 101):
        dist = galton_board_distribution_risk(steps, risk)
        color = cmap(risk / 100)
        plt.plot(x, dist, color=color, alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=100))
    cbar = plt.colorbar(sm, label='Risk (% chance to go down)', ax=plt.gca())
    plt.xlabel('Number of Ups')
    plt.ylabel('Probability')
    plt.title(f'Galton Board Distributions Overlayed ({steps} steps, Risk 1-100%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show() 