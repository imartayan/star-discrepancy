import ioh
import warnings
import numpy as np
import heuristics as hr
from statistics import mean


BUDGET = 50
N_PTS = 5
THRESHOLD = 1

algos = {
    "Random": hr.RandomSearch(BUDGET),
    "BruteForce": hr.BruteForce(BUDGET),
    "GaussianLS (.3)": hr.GaussianLocalSearch(0.3, BUDGET),
    "GaussianLS (.2)": hr.GaussianLocalSearch(0.2, BUDGET),
    "GaussianLS (.1)": hr.GaussianLocalSearch(0.1, BUDGET),
    "GaussianLS (.05)": hr.GaussianLocalSearch(0.05, BUDGET),
    # "GaussianSingleAxis  (.3)": hr.GaussianSingleAxis(.3, BUDGET),
    # "GaussianSingleAxis  (.2)": hr.GaussianSingleAxis(.2, BUDGET),
    # "GaussianSingleAxis  (.1)": hr.GaussianSingleAxis(.1, BUDGET),
    # "GaussianSingleAxis  (.05)": hr.GaussianSingleAxis(.05, BUDGET),
    # "ExpLS (.3)": hr.ExpLocalSearch(.3, BUDGET),
    "ExpLS (.3)": hr.ExpLocalSearch(0.3, BUDGET),
    "ExpLS (.2)": hr.ExpLocalSearch(0.2, BUDGET),
    "ExpLS (.1)": hr.ExpLocalSearch(0.1, BUDGET),
    "ExpLS (.05)": hr.ExpLocalSearch(0.05, BUDGET),
    # "ExpLS (.03)": hr.ExpLocalSearch(.03, BUDGET),
    # "GaussianCrossover  (.1)": hr.GaussianCrossover(.1, BUDGET),
    # "GaussianCrossover  (.05)": hr.GaussianCrossover(.05, BUDGET),
    # "GaussianCrossover  (.03)": hr.GaussianCrossover(.03, BUDGET),
    # "GaussianReset (.3)": hr.GaussianReset(.3, THRESHOLD, BUDGET),
    # "GaussianReset (.2)": hr.GaussianReset(.2, THRESHOLD, BUDGET),
    # "GaussianReset (.1)": hr.GaussianReset(.1, THRESHOLD, BUDGET),
    # "GaussianReset (.05)": hr.GaussianReset(.05, THRESHOLD, BUDGET),
    # "GaussianMLS (.3)": hr.GaussianMultiLocalSearch(.3, N_PTS, BUDGET),
    "GaussianMLS (.2)": hr.GaussianMultiLocalSearch(0.2, N_PTS, BUDGET),
    "GaussianMLS (.1)": hr.GaussianMultiLocalSearch(0.1, N_PTS, BUDGET),
    "GaussianMLS (.05)": hr.GaussianMultiLocalSearch(0.05, N_PTS, BUDGET),
    "GaussianMLS (.03)": hr.GaussianMultiLocalSearch(0.03, N_PTS, BUDGET),
    "GaussianMLS (.02)": hr.GaussianMultiLocalSearch(0.02, N_PTS, BUDGET),
    # "ExpMLS (.3)": hr.ExpMultiLocalSearch(.3, N_PTS, BUDGET),
    "ExpMLS (.2)": hr.ExpMultiLocalSearch(0.2, N_PTS, BUDGET),
    "ExpMLS (.1)": hr.ExpMultiLocalSearch(0.1, N_PTS, BUDGET),
    "ExpMLS (.05)": hr.ExpMultiLocalSearch(0.05, N_PTS, BUDGET),
    "ExpMLS (.03)": hr.ExpMultiLocalSearch(0.03, N_PTS, BUDGET),
    "ExpMLS (.02)": hr.ExpMultiLocalSearch(0.02, N_PTS, BUDGET),
    # "GaussianWMLS (.3)": hr.GaussianWeightedMultiLocalSearch(.3, N_PTS, BUDGET),
    # "GaussianWMLS (.2)": hr.GaussianWeightedMultiLocalSearch(.2, N_PTS, BUDGET),
    # "GaussianWMLS (.1)": hr.GaussianWeightedMultiLocalSearch(.1, N_PTS, BUDGET),
    # "GaussianWMLS (.05)": hr.GaussianWeightedMultiLocalSearch(.05, N_PTS, BUDGET),
    # # "ExpWMLS (.3)": hr.ExpWeightedMultiLocalSearch(.3, N_PTS, BUDGET),
    # "ExpWMLS (.2)": hr.ExpWeightedMultiLocalSearch(.2, N_PTS, BUDGET),
    # "ExpWMLS (.1)": hr.ExpWeightedMultiLocalSearch(.1, N_PTS, BUDGET),
    # "ExpWMLS (.05)": hr.ExpWeightedMultiLocalSearch(.05, N_PTS, BUDGET),
    # "ExpWMLS (.03)": hr.ExpWeightedMultiLocalSearch(.03, N_PTS, BUDGET),
    "GaussianALS (.5)": hr.GaussianAdaptiveLocalSearch(0.5, BUDGET),
    "GaussianALS (.3)": hr.GaussianAdaptiveLocalSearch(0.3, BUDGET),
    "GaussianALS (.2)": hr.GaussianAdaptiveLocalSearch(0.2, BUDGET),
    # "GaussianALS (.1)": hr.GaussianAdaptiveLocalSearch(.1, BUDGET),
    "ExpALS (.5)": hr.ExpAdaptiveLocalSearch(0.5, BUDGET),
    "ExpALS (.3)": hr.ExpAdaptiveLocalSearch(0.3, BUDGET),
    "ExpALS (.2)": hr.ExpAdaptiveLocalSearch(0.2, BUDGET),
    # "ExpALS (.1)": hr.ExpAdaptiveLocalSearch(.1, BUDGET),
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
    fid = 32
    dim = 5
    benchmark(fid, dim)
