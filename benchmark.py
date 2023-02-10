import ioh
import numpy as np
import heuristics as hr
from statistics import mean, stdev


BUDGET = 100
N_PTS = 5
STEPS = [0.2, 0.1]

algos = (
    {
        "Random": hr.RandomSearch(BUDGET),
        "BruteForce": hr.BruteForce(BUDGET),
    }
    | {f"GaussianSingle ({s})": hr.GaussianSingleAxis(s, BUDGET) for s in STEPS}
    | {f"GaussianLS ({s})": hr.GaussianLocalSearch(s, BUDGET) for s in STEPS}
    | {f"GaussianMLS ({s})": hr.GaussianLocalSearch(s, BUDGET, N_PTS) for s in STEPS}
    | {f"GaussianLSR ({s}, 10)": hr.GaussianReset(s, 10, BUDGET) for s in STEPS}
    | {f"GaussianMLSR ({s}, 10)": hr.GaussianReset(s, 10, BUDGET, N_PTS) for s in STEPS}
    # | {
    #     f"GaussianALS ({s})": hr.AltGaussianAdaptiveLocalSearch(s, BUDGET)
    #     for s in STEPS
    # }
    | {f"ExpLS ({s})": hr.ExpLocalSearch(s, BUDGET) for s in STEPS}
    | {f"ExpMLS ({s})": hr.ExpLocalSearch(s, BUDGET, N_PTS) for s in STEPS}
    | {
        f"ExpSA ({s}, 1, 0.99)": hr.ExpSimulatedAnnealing(s, 1, 0.99, BUDGET)
        for s in STEPS
    }
    | {
        f"ExpSA ({s}, 0.5, 0.99)": hr.ExpSimulatedAnnealing(s, 0.5, 0.99, BUDGET)
        for s in STEPS
    }
)


def benchmark(fid: int, dim: int, rep: int = 10):
    for name, algo in algos.items():
        func = ioh.get_problem(
            fid, dimension=dim, instance=1, problem_type=ioh.ProblemType.REAL
        )
        res = []
        for r in range(rep):
            np.random.seed(r)
            algo(func)
            y = func.state.current_best.y
            res.append(y)
            func.reset()
        print(f"{name}\t{'%.3e' % mean(res)} ± {'%.1e' % stdev(res)}")


if __name__ == "__main__":
    fid = 33
    dim = 3
    benchmark(fid, dim)
