import ioh
import numpy as np


def gaussian_step(x, avg_step):
    sigma = avg_step * np.sqrt(np.pi / 2)
    y = x + np.random.normal(scale=sigma, size=len(x))
    return np.clip(y, 0, 1)


def exp_step(x, avg_step):
    beta = avg_step
    sign = np.random.choice([-1, 1], size=len(x))
    step = np.random.exponential(scale=beta, size=len(x))
    y = x + sign * step
    return np.clip(y, 0, 1)


def biased_exp_step(x, avg_step):
    beta = avg_step
    if np.random.randint(2) == 0:
        y = x + np.random.exponential(scale=beta, size=len(x))
    else:
        y = x - np.random.exponential(scale=beta, size=len(x))
    return np.clip(y, 0, 1)


"""
Naive Heuristics
"""


class Search:
    def __init__(self, budget_factor):
        self.budget_factor = budget_factor


class RandomSearch(Search):
    def __call__(self, func: ioh.problem.RealSingleObjective):
        n = func.meta_data.n_variables
        d = len(func.bounds.lb)
        n_calls = self.budget_factor * n * d
        for _ in range(n_calls):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            func(x)
        return func.state.current_best


class BruteForce(Search):
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


class LocalSearch(Search):
    def __init__(self, avg_step, budget_factor):
        super().__init__(budget_factor=budget_factor)
        self.avg_step = avg_step

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
                fx = fy
        return func.state.current_best


class GaussianSingleAxis(LocalSearch):
    def local_step(self, x):
        k = np.random.randint(len(x))
        y = x[:]
        y[k] += np.random.normal(scale=self.sigma)
        return np.clip(y, 0, 1)


class GaussianLocalSearch(LocalSearch):
    def local_step(self, x):
        return gaussian_step(x, self.avg_step)


class ExpLocalSearch(LocalSearch):
    def local_step(self, x):
        return exp_step(x, self.avg_step)


# class TrueExpLocalSearch(LocalSearch):
#     def local_step(self, x):
#         return exp_step(x, self.avg_step)


"""
Local Search with Resets
"""


class GaussianReset(GaussianLocalSearch):
    def __init__(self, avg_step, threshold_factor, budget_factor):
        super().__init__(avg_step=avg_step, budget_factor=budget_factor)
        self.threshold_factor = threshold_factor

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
                fx = fy
                rep = 0
            else:
                rep += 1
        return func.state.current_best


"""
Multiple Local Search at once
"""


class MultiLocalSearch(LocalSearch):
    def __init__(self, avg_step, n_points, budget_factor):
        super().__init__(avg_step=avg_step, budget_factor=budget_factor)
        self.n_points = n_points

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
    def local_step(self, x):
        return gaussian_step(x, self.avg_step)


class ExpMultiLocalSearch(MultiLocalSearch):
    def local_step(self, x):
        return exp_step(x, self.avg_step)


"""
Weighted Multiple Local Search at once
"""


class WeightedMultiLocalSearch(MultiLocalSearch):
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
        indices = np.array(range(self.n_points))
        scores = fxs / fxs.sum()
        weights = np.exp(scores)
        weights /= weights.sum()
        for _ in range(n_calls):
            i = np.random.choice(indices, p=weights)
            y = self.local_step(xs[i])
            fy = func(y)
            if fy > fxs[i]:
                xs[i] = y
                fxs[i] = fy
                scores = fxs / fxs.sum()
                weights = np.exp(scores)
                weights /= weights.sum()
        return func.state.current_best


class GaussianWeightedMultiLocalSearch(WeightedMultiLocalSearch):
    def local_step(self, x):
        return gaussian_step(x, self.avg_step)


class ExpWeightedMultiLocalSearch(WeightedMultiLocalSearch):
    def local_step(self, x):
        return exp_step(x, self.avg_step)


"""
Adaptive Local Search
"""


class AdaptiveLocalSearch(LocalSearch):
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
                avg_step = np.mean(np.abs(y - x))
                alpha = 0.995
                self.avg_step = alpha * self.avg_step + (1 - alpha) * avg_step
                x = y
                fx = fy
        return func.state.current_best


class GaussianAdaptiveLocalSearch(AdaptiveLocalSearch):
    def local_step(self, x):
        return gaussian_step(x, self.avg_step)


class ExpAdaptiveLocalSearch(AdaptiveLocalSearch):
    def local_step(self, x):
        return exp_step(x, self.avg_step)


"""
Local Search with Crossover (not good yet)
"""


class GaussianCrossover(LocalSearch):
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
            if np.random.randint(2) == 0:
                y = self.upstep(x0, x1)
                fy = func(y)
                if fy > fx0:
                    x0 = y
                    fx0 = fy
            else:
                y = self.downstep(x0, x1)
                fy = func(y)
                if fy > fx1:
                    x1 = y
                    fx1 = y
        return func.state.current_best
