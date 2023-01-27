import ioh
import numpy as np
from math import sqrt


class BruteForce:
    def __init__(self, budget_factor: int = 1000):
        self.budget_factor = budget_factor

    def __call__(self, func: ioh.problem.RealSingleObjective):
        n_calls = self.budget_factor * func.meta_data.n_variables
        x = np.zeros(2)
        N = int(sqrt(n_calls))
        for i in range(N):
            for j in range(N):
                x[0] = i / N
                x[1] = j / N
                func(x)
        return func.state.current_best


class RandomSearch:
    def __init__(self, budget_factor: int = 1000):
        self.budget_factor = budget_factor

    def __call__(self, func: ioh.problem.RealSingleObjective):
        n_calls = self.budget_factor * func.meta_data.n_variables
        for _ in range(n_calls):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            func(x)
        return func.state.current_best


class GaussianLocalSearch:
    def __init__(self, scale, budget_factor: int = 1000):
        self.budget_factor = budget_factor
        self.scale = scale

    def local_step(self, x):
        y = x + np.random.normal(scale=self.scale, size=len(x))
        return np.clip(y, 0, 1)

    def __call__(self, func: ioh.problem.RealSingleObjective):
        n_calls = self.budget_factor * func.meta_data.n_variables
        x = np.random.uniform(func.bounds.lb, func.bounds.ub)
        fx = func(x)
        for _ in range(n_calls):
            y = self.local_step(x)
            fy = func(y)
            if fy > fx:
                x = y
        return func.state.current_best


class GaussianSingleAxis(GaussianLocalSearch):
    def local_step(self, x):
        k = np.random.randint(len(x))
        y = x[:]
        y[k] += np.random.normal(scale=self.scale)
        return np.clip(y, 0, 1)


all = {
    "BruteForce": BruteForce(),
    "RandomSearch": RandomSearch(),
    "GaussianLocalSearch (0.3)": GaussianLocalSearch(0.3),
    "GaussianLocalSearch (0.2)": GaussianLocalSearch(0.2),
    "GaussianLocalSearch (0.1)": GaussianLocalSearch(0.1),
    "GaussianLocalSearch (0.05)": GaussianLocalSearch(0.05),
    "GaussianSingleAxis  (0.3)": GaussianSingleAxis(0.3),
    "GaussianSingleAxis  (0.2)": GaussianSingleAxis(0.2),
    "GaussianSingleAxis  (0.1)": GaussianSingleAxis(0.1),
    "GaussianSingleAxis  (0.05)": GaussianSingleAxis(0.05),
}
