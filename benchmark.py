import ioh
import warnings
import numpy as np
import heuristics as hr
from statistics import mean


BUDGET = 50
N_PTS = 5
STEPS = [0.2, 0.1, 0.05]

algos = (
    {
        "Random": hr.RandomSearch(BUDGET),
        "BruteForce": hr.BruteForce(BUDGET),
    }
    | {f"GaussianLS ({s})": hr.GaussianLocalSearch(s, BUDGET) for s in STEPS}
    | {f"ExpLS ({s})": hr.ExpLocalSearch(s, BUDGET) for s in STEPS}
    | {
        f"GaussianMLS ({s})": hr.GaussianMultiLocalSearch(s, N_PTS, BUDGET)
        for s in STEPS
    }
    | {f"ExpMLS ({s})": hr.ExpMultiLocalSearch(s, N_PTS, BUDGET) for s in STEPS}
    | {
        f"ExpSA ({s}, 1, 0.99)": hr.ExpSimulatedAnnealing(s, 1, 0.99, BUDGET)
        for s in STEPS
    }
    | {
        f"ExpSA ({s}, 0.5, 0.99)": hr.ExpSimulatedAnnealing(s, 0.5, 0.99, BUDGET)
        for s in STEPS
    }
)


def benchmark(fid: int, dim: int):
    for name, algo in algos.items():
        means = []
        bests = []
        for iid in range(5):
            func = ioh.get_problem(
                fid, dimension=dim, instance=iid, problem_type=ioh.ProblemType.REAL
            )
            res = []
            for rep in range(10):
                np.random.seed(rep)
                algo(func)
                y = func.state.current_best.y
                res.append(y)
                func.reset()
            means.append(mean(res))
            bests.append(max(res))
        print(f"{name}\t{'%.3e' % mean(means)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    fid = 52
    dim = 4
    benchmark(fid, dim)
