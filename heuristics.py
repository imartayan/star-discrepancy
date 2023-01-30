import ioh
import numpy as np

DEFAULT_BUGDET_FACTOR = 100


"""
Naive Heuristics
"""


class RandomSearch:
    def __init__(self, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor

    def __call__(self, func: ioh.problem.RealSingleObjective):
        n = func.meta_data.n_variables
        d = len(func.bounds.lb)
        n_calls = self.budget_factor * n * d
        for _ in range(n_calls):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            func(x)
        return func.state.current_best


class BruteForce:
    def __init__(self, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor

    def __call__(self, func: ioh.problem.RealSingleObjective):
        n = func.meta_data.n_variables
        d = len(func.bounds.lb)
        n_calls = self.budget_factor * n * d
        N = int(n_calls ** (1 / d))
        axis = [np.linspace(0, 1, N) for _ in range(d)]
        mesh = np.meshgrid(*axis)
        points = np.stack(mesh, axis=-1)
        for x in points.reshape((N**d, d)):
            func(x)
        return func.state.current_best


"""
Local Search
"""


class LocalSearch:
    def __init__(self, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor

    def local_step(self, x):
        raise NotImplementedError

    def __call__(self, func: ioh.problem.RealSingleObjective):
        n = func.meta_data.n_variables
        d = len(func.bounds.lb)
        n_calls = self.budget_factor * n * d
        x = np.random.uniform(func.bounds.lb, func.bounds.ub)
        fx = func(x)
        for _ in range(n_calls):
            y = self.local_step(x)
            fy = func(y)
            if fy > fx:
                x = y
        return func.state.current_best


class GaussianLocalSearch(LocalSearch):
    def __init__(self, sigma, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor
        self.sigma = sigma

    def local_step(self, x):
        y = x + np.random.normal(scale=self.sigma, size=len(x))
        return np.clip(y, 0, 1)


class GaussianSingleAxis(GaussianLocalSearch):
    def local_step(self, x):
        k = np.random.randint(len(x))
        y = x[:]
        y[k] += np.random.normal(scale=self.sigma)
        return np.clip(y, 0, 1)


class ExpLocalSearch(LocalSearch):
    def __init__(self, beta, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor
        self.beta = beta

    def local_step(self, x):
        if np.random.random() < 0.5:
            y = x + np.random.exponential(scale=self.beta, size=len(x))
        else:
            y = x - np.random.exponential(scale=self.beta, size=len(x))
        return np.clip(y, 0, 1)


"""
Local Search with Resets
"""


class GaussianReset:
    def __init__(self, sigma, threshold_factor=1, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor
        self.threshold_factor = threshold_factor
        self.sigma = sigma

    def local_step(self, x):
        y = x + np.random.normal(scale=self.sigma, size=len(x))
        return np.clip(y, 0, 1)

    def __call__(self, func: ioh.problem.RealSingleObjective):
        n = func.meta_data.n_variables
        d = len(func.bounds.lb)
        n_calls = self.budget_factor * n * d
        x = np.random.uniform(func.bounds.lb, func.bounds.ub)
        fx = func(x)
        rep = 0
        threshold = self.threshold_factor * d
        for _ in range(n_calls):
            if rep == threshold:
                x = np.random.uniform(func.bounds.lb, func.bounds.ub)
                fx = func(x)
                rep = 0
            y = self.local_step(x)
            fy = func(y)
            if fy > fx:
                x = y
                rep = 0
            else:
                rep += 1
        return func.state.current_best


"""
Multiple Local Search at once
"""


class MultiLocalSearch:
    def __init__(self, n_points=5, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor
        self.n_points = n_points

    def local_step(self, x):
        raise NotImplementedError

    def __call__(self, func: ioh.problem.RealSingleObjective):
        n = func.meta_data.n_variables
        d = len(func.bounds.lb)
        n_calls = self.budget_factor * n * d
        xs = np.array(
            [
                np.random.uniform(func.bounds.lb, func.bounds.ub)
                for _ in range(self.n_points)
            ]
        )
        fxs = np.array([func(x) for x in xs])
        for _ in range(n_calls):
            i = np.random.randint(len(xs))
            y = self.local_step(xs[i])
            fy = func(y)
            if fy > fxs[i]:
                xs[i] = y
                fxs[i] = fy
        return func.state.current_best


class GaussianMultiLocalSearch(MultiLocalSearch):
    def __init__(self, sigma, n_points=5, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor
        self.sigma = sigma
        self.n_points = n_points

    def local_step(self, x):
        y = x + np.random.normal(scale=self.sigma, size=len(x))
        return np.clip(y, 0, 1)


class ExpMultiLocalSearch(MultiLocalSearch):
    def __init__(self, beta, n_points=5, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor
        self.beta = beta
        self.n_points = n_points

    def local_step(self, x):
        if np.random.random() < 0.5:
            y = x + np.random.exponential(scale=self.beta, size=len(x))
        else:
            y = x - np.random.exponential(scale=self.beta, size=len(x))
        return np.clip(y, 0, 1)


"""
Weighted Multiple Local Search at once
"""


class WeightedMultiLocalSearch:
    def __init__(self, n_points=5, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor
        self.n_points = n_points

    def local_step(self, x):
        raise NotImplementedError

    def __call__(self, func: ioh.problem.RealSingleObjective):
        n = func.meta_data.n_variables
        d = len(func.bounds.lb)
        n_calls = self.budget_factor * n * d
        xs = np.array(
            [
                np.random.uniform(func.bounds.lb, func.bounds.ub)
                for _ in range(self.n_points)
            ]
        )
        fxs = np.array([func(x) for x in xs])
        fsum = fxs.sum()
        weights = fxs / fsum
        weights = np.exp(weights)
        weights /= weights.sum()
        for _ in range(n_calls):
            i = np.random.choice(list(range(len(xs))), p=weights)
            y = self.local_step(xs[i])
            fy = func(y)
            if fy > fxs[i]:
                xs[i] = y
                fxs[i] = fy
                fsum = fxs.sum()
                weights = fxs / fsum
                weights = np.exp(weights)
                weights /= weights.sum()
        return func.state.current_best


class GaussianWeightedMultiLocalSearch(WeightedMultiLocalSearch):
    def __init__(self, sigma, n_points=5, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor
        self.sigma = sigma
        self.n_points = n_points

    def local_step(self, x):
        y = x + np.random.normal(scale=self.sigma, size=len(x))
        return np.clip(y, 0, 1)


class ExpWeightedMultiLocalSearch(WeightedMultiLocalSearch):
    def __init__(self, beta, n_points=5, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor
        self.beta = beta
        self.n_points = n_points

    def local_step(self, x):
        if np.random.random() < 0.5:
            y = x + np.random.exponential(scale=self.beta, size=len(x))
        else:
            y = x - np.random.exponential(scale=self.beta, size=len(x))
        return np.clip(y, 0, 1)


"""
Local Search with Crossover (not good yet)
"""


class GaussianCrossover:
    # FIXME

    def __init__(self, sigma, budget_factor=DEFAULT_BUGDET_FACTOR):
        self.budget_factor = budget_factor
        self.sigma = sigma

    def upstep(self, x0, x1):
        axis = [i for i, (u, v) in enumerate(zip(x0, x1)) if u < v]
        y = x0[:]
        y[axis] += np.random.normal(scale=self.sigma, size=len(axis))
        return np.clip(y, 0, 1)

    def downstep(self, x0, x1):
        axis = [i for i, (u, v) in enumerate(zip(x0, x1)) if u < v]
        y = x1[:]
        y[axis] += np.random.normal(scale=self.sigma, size=len(axis))
        return np.clip(y, 0, 1)

    def __call__(self, func: ioh.problem.RealSingleObjective):
        n = func.meta_data.n_variables
        d = len(func.bounds.lb)
        n_calls = self.budget_factor * n * d
        d = len(func.bounds.lb)
        x0 = np.zeros(d)
        x1 = np.ones(d)
        fx0 = func(x0)
        fx1 = func(x1)
        for _ in range(n_calls):
            if np.random.random() < 0.5:
                y = self.upstep(x0, x1)
                fy = func(y)
                if fy > fx0:
                    x0 = y
            else:
                y = self.downstep(x0, x1)
                fy = func(y)
                if fy > fx1:
                    x1 = y
        return func.state.current_best
