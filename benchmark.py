import ioh
import warnings
import numpy as np
import heuristics
from statistics import mean


def benchmark(fid: int, dim: int):
    for name, algo in heuristics.all.items():
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
        print(f"{name}\t{'%.2e' % mean(means)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    fid = 39
    dim = 2
    benchmark(fid, dim)
