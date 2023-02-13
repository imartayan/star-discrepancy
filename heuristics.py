import ioh
import numpy as np


def uniform_step(x, avg_step):
    amp = 2 * avg_step
    y = -1
    while np.any(y > 1) or np.any(y < 0):
        y = x + np.random.uniform(-amp, amp, size=len(x))
    return y


def gaussian_step(x, avg_step):
    sigma = avg_step * np.sqrt(np.pi / 2)
    y = -1
    while np.any(y > 1) or np.any(y < 0):
        y = x + np.random.normal(scale=sigma, size=len(x))
    return y


def exp_step(x, avg_step):
    beta = avg_step
    y = -1
    while np.any(y > 1) or np.any(y < 0):
        sign = np.random.choice([-1, 1], size=len(x))
        step = np.random.exponential(scale=beta, size=len(x))
        y = x + sign * step
    return y


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
    def __init__(self, avg_step, budget_factor, n_points=1):
        super().__init__(budget_factor=budget_factor)
        self.avg_step = avg_step
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
        for c in range(n_calls):
            i = c % self.n_points
            y = self.local_step(xs[i])
            fy = func(y)
            if fy > fxs[i]:
                xs[i] = y
                fxs[i] = fy
        return func.state.current_best


class UniformLocalSearch(LocalSearch):
    def local_step(self, x):
        return uniform_step(x, self.avg_step)


class GaussianLocalSearch(LocalSearch):
    def local_step(self, x):
        return gaussian_step(x, self.avg_step)


class ExpLocalSearch(LocalSearch):
    def local_step(self, x):
        return exp_step(x, self.avg_step)


"""
Local Search with Resets
"""


class LocalSearchReset(LocalSearch):
    def __init__(self, avg_step, threshold_factor, budget_factor, n_points=1):
        super().__init__(
            avg_step=avg_step, budget_factor=budget_factor, n_points=n_points
        )
        self.threshold_factor = threshold_factor

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
        stuck = np.zeros(self.n_points)
        threshold = self.threshold_factor * d
        for c in range(n_calls):
            i = c % self.n_points
            if stuck[i] == threshold:
                xs[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                fxs[i] = func(xs[i])
                stuck[i] = 0
            y = self.local_step(xs[i])
            fy = func(y)
            if fy > fxs[i]:
                xs[i] = y
                fxs[i] = fy
                stuck[i] = 0
            else:
                stuck[i] += 1
        return func.state.current_best


class GaussianLocalSearchReset(LocalSearchReset):
    def local_step(self, x):
        return gaussian_step(x, self.avg_step)


class ExpLocalSearchReset(LocalSearchReset):
    def local_step(self, x):
        return exp_step(x, self.avg_step)


"""
Weighted Local Search
"""


class WeightedLocalSearch(LocalSearch):
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


class GaussianWeightedLocalSearch(WeightedLocalSearch):
    def local_step(self, x):
        return gaussian_step(x, self.avg_step)


class ExpWeightedLocalSearch(WeightedLocalSearch):
    def local_step(self, x):
        return exp_step(x, self.avg_step)


"""
Adaptive Local Search (bad)
"""


class AdaptiveLocalSearch(LocalSearch):
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
        for c in range(n_calls):
            i = c % self.n_points
            y = self.local_step(xs[i])
            fy = func(y)
            if fy > fxs[i]:
                avg_step = np.mean(np.abs(y - xs[i]))
                alpha = 0.8
                self.avg_step = alpha * self.avg_step + (1 - alpha) * avg_step
                xs[i] = y
                fxs[i] = fy
        return func.state.current_best


"""
Simulated Annealing
"""


class SimulatedAnnealing(LocalSearch):
    def __init__(self, avg_step, temp_init, temp_decay, budget_factor, n_points=1):
        super().__init__(
            avg_step=avg_step, budget_factor=budget_factor, n_points=n_points
        )
        self.temp_init = temp_init
        self.temp_decay = temp_decay

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
        temp = self.temp_init
        for c in range(n_calls):
            i = c % self.n_points
            y = self.local_step(xs[i])
            fy = func(y)
            delta = fy - fxs[i]
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                xs[i] = y
                fxs[i] = fy
            temp *= self.temp_decay
        return func.state.current_best


class GaussianSimulatedAnnealing(SimulatedAnnealing):
    def local_step(self, x):
        return gaussian_step(x, self.avg_step)


class ExpSimulatedAnnealing(SimulatedAnnealing):
    def local_step(self, x):
        return exp_step(x, self.avg_step)
