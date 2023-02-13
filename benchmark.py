import ioh
import numpy as np
import heuristics as hr
import matplotlib.pyplot as plt
from statistics import mean, stdev


def benchmark(algos, fid, dim, rep):
    func = ioh.get_problem(
        fid, dimension=dim, instance=1, problem_type=ioh.ProblemType.REAL
    )
    stats = {}
    for name, algo in algos.items():
        res = []
        for r in range(rep):
            np.random.seed(r)
            algo(func)
            y = func.state.current_best.y
            res.append(y)
            func.reset()
        stats[name] = (mean(res), stdev(res) / np.sqrt(rep))
    return stats


def get_algos(budget, steps, n_points):
    return (
        {}
        | {"Random Search": hr.RandomSearch(budget)}
        # | {"Brute Force": hr.BruteForce(budget)}
        # | {f"uniform ({s})": hr.UniformLocalSearch(s, budget) for s in steps}
        # | {f"gaussian ({s})": hr.GaussianLocalSearch(s, budget) for s in steps}
        # | {f"exponential ({s})": hr.ExpLocalSearch(s, budget) for s in steps}
        # | {
        #     f"exponential ({s}, {n_points})": hr.ExpLocalSearch(s, budget, n_points)
        #     for s in steps
        # }
        # | {
        #     f"exponential ({s}, {2*n_points})": hr.ExpLocalSearch(
        #         s, budget, 2 * n_points
        #     )
        #     for s in steps
        # }
        # | {
        #     f"exponential ({s}, {4*n_points})": hr.ExpLocalSearch(
        #         s, budget, 4 * n_points
        #     )
        #     for s in steps
        # }
        # | {f"exponential reset ({s}, 10)": hr.ExpReset(s, 10, budget) for s in steps}
        | {
            f"ESA ({s}, 0.003, 0.999)": hr.ExpSimulatedAnnealing(
                s, 0.003, 0.999, budget
            )
            for s in steps
        }
        | {
            f"ESA ({s}, 0.005, 0.997)": hr.ExpSimulatedAnnealing(
                s, 0.005, 0.997, budget
            )
            for s in steps
        }
    )


def print_table(scores):
    print("table ====================")
    for name, val in scores.items():
        print(f"{name}", end="")
        for v in val:
            print(f" & {v}", end="")
        print("\\\\")


def plot_chart(scores, save=None):
    n_algos = len(scores)
    W = 0.7
    w = W / n_algos
    offset = np.linspace(-W / 2, W / 2, n_algos)
    rects = []

    fig, ax = plt.subplots()
    for i, (a, m) in enumerate(scores.items()):
        x = np.arange(len(m))
        rects.append(ax.bar(x + offset[i], m, w, label=a))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel("Scores")
    # ax.set_title("Scores by group and gender")
    # ax.set_xticks(x, labels)
    ax.legend()

    for rect in rects:
        ax.bar_label(rect)
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    if save:
        pass
    else:
        plt.show()


if __name__ == "__main__":
    dim = 3
    rep = 200
    n_points = 2
    steps = [0.1]
    scores = {}
    for fid in [33, 43, 53]:
        print(f"PROBLEM {fid} ====================")
        for budget in [20, 200]:
            print(f"BUDGET {budget} --------------------")
            stats = benchmark(get_algos(budget, steps, n_points), fid, dim, rep)
            for name, (m, e) in stats.items():
                print(f"{name}\t{'%.3e' % m} ± {'%.0e' % e}")
                if name not in scores:
                    scores[name] = []
                scores[name].append(m)
    print_table(scores)
    # plot_chart(scores)
