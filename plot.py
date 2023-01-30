import matplotlib.pyplot as plt
from math import prod
import random as rd
import numpy as np


def discrepancy(X, y):
    vol = prod(y)
    inside = [x for x in X if all(xi < yi for xi, yi in zip(x, y))]
    return vol - len(inside) / len(X)


def plot_1d(X, save=None):
    fig, ax = plt.subplots()
    n = len(X)
    Y = [0]
    D = [0]
    for i, x in enumerate(X):
        Y.append(x)
        D.append(x - i / n)
        Y.append(x)
        D.append(x - (i + 1) / n)
    Y.append(1)
    D.append(0)
    ax.plot([0, 1], [0, 0], "r--")
    ax.plot(Y, D)
    ax.plot(X, [0 for _ in X], marker="*", color="black", linestyle="")
    ax.set_xlabel("y")
    ax.set_ylabel(r"vol(y) - $\frac{|X \cap [0,y]|}{|X|}$")
    fig.tight_layout()
    if save:
        plt.savefig(save)
    else:
        plt.show()


def plot_2d(X, save=None, N=200):
    u, v = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    Y = np.dstack((u, v))
    D = np.apply_along_axis(lambda y: abs(discrepancy(X, y)), 2, Y)
    fig, ax = plt.subplots()
    im = ax.imshow(D, origin="lower", cmap="coolwarm")
    ax.axis("off")
    fig.colorbar(im, ax=ax)
    X0 = [x[0] * N for x in X]
    X1 = [x[1] * N for x in X]
    ax.plot(X0, X1, marker="*", color="black", linestyle="")
    fig.tight_layout()
    if save:
        plt.savefig(save)
    else:
        plt.show()


if __name__ == "__main__":
    X1 = sorted([rd.uniform(0.1, 0.9) for _ in range(5)])
    # plot_1d(X1, "plot_1d.pdf")
    X2 = [(rd.uniform(0.1, 0.9), rd.uniform(0.1, 0.9)) for _ in range(5)]
    # plot_2d(X2, "plot_2d.pdf")
