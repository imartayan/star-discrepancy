import ioh
import warnings
import numpy as np
import heuristics
from statistics import mean


algos = {
    "RandomSearch": heuristics.RandomSearch(),
    "BruteForce": heuristics.BruteForce(),
    "GaussianLS (0.3)": heuristics.GaussianLocalSearch(0.3),
    "GaussianLS (0.2)": heuristics.GaussianLocalSearch(0.2),
    "GaussianLS (0.1)": heuristics.GaussianLocalSearch(0.1),
    "GaussianLS (0.05)": heuristics.GaussianLocalSearch(0.05),
    "GaussianLS (0.03)": heuristics.GaussianLocalSearch(0.03),
    "GaussianLS (0.01)": heuristics.GaussianLocalSearch(0.01),
    # "GaussianSingleAxis  (0.3)": heuristics.GaussianSingleAxis(0.3),
    # "GaussianSingleAxis  (0.2)": heuristics.GaussianSingleAxis(0.2),
    # "GaussianSingleAxis  (0.1)": heuristics.GaussianSingleAxis(0.1),
    # "GaussianSingleAxis  (0.05)": heuristics.GaussianSingleAxis(0.05),
    "ExpLS (0.3)": heuristics.ExpLocalSearch(0.3),
    "ExpLS (0.2)": heuristics.ExpLocalSearch(0.2),
    "ExpLS (0.1)": heuristics.ExpLocalSearch(0.1),
    "ExpLS (0.05)": heuristics.ExpLocalSearch(0.05),
    "ExpLS (0.03)": heuristics.ExpLocalSearch(0.03),
    "ExpLS (0.01)": heuristics.ExpLocalSearch(0.01),
    "GaussianReset (0.3)": heuristics.GaussianReset(0.3),
    "GaussianReset (0.2)": heuristics.GaussianReset(0.2),
    "GaussianReset (0.1)": heuristics.GaussianReset(0.1),
    "GaussianReset (0.05)": heuristics.GaussianReset(0.05),
    "GaussianReset (0.03)": heuristics.GaussianReset(0.03),
    "GaussianReset (0.01)": heuristics.GaussianReset(0.01),
    # "GaussianCrossover  (0.1)": heuristics.GaussianCrossover(0.1),
    # "GaussianCrossover  (0.05)": heuristics.GaussianCrossover(0.05),
    # "GaussianCrossover  (0.03)": heuristics.GaussianCrossover(0.03),
}


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
    fid = 30
    dim = 3
    benchmark(fid, dim)
