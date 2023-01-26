import numpy as np
import ioh  # Make sure to use version >= 0.3.6

# import sys
# import argparse
# import os
import warnings


class RandomSearch:
    """An example of Random Search."""

    def __init__(self, budget_factor: int = 1000):
        self.budget_factor = budget_factor

    def __call__(self, func: ioh.problem.RealSingleObjective):
        for _ in range(self.budget_factor * func.meta_data.n_variables):
            # Generate a random point uniformly at random
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            # Evaluate it
            y = func(x)
        # We don't have to explicitly track the best-so-far, the problem's internal state takes care of that
        return func.state.current_best


def run_randomsearch(fid: int, dim: int, log_dir: str = None, verbose: bool = True):
    """An example of using the created Random Search on a star-discrepancy problem

    This function takes care of creating the problem and logger (if needed)
    and performs a basic experiment consisting of 10 runs on 5 instances of the selected problem.

    Parameters
    ----------
    fid: int
        The function number (between 30 and 59 for the star-discrepancy problems).
    dim: int
        The dimension (number of variables) of the problem.
    log_dir: str
        Where to store the data (if required). If set to None, no data will be recorded.
    verbose: bool
        Whether or not to print the final result of each run.
    """

    # Make the algorithm and set how we want it to be called in the logs
    algname = "RandomSearch"
    algorithm = RandomSearch()

    # If we want to store data, we need to make a logger
    if log_dir is not None:
        logger = ioh.logger.Analyzer(
            root=log_dir, folder_name=f"F{fid}_{dim}D", algorithm_name=f"{algname}"
        )

    # Loop over the instances, and create the problems
    for iid in range(5):
        func = ioh.get_problem(
            fid, dimension=dim, instance=iid, problem_type=ioh.ProblemType.REAL
        )

        # Print some info on the current problem if needed
        if verbose:
            print(func)

        if log_dir is not None:
            func.attach_logger(logger)
        for rep in range(10):
            # Set the seed for reproducibility
            np.random.seed(rep)
            # Run the algorithm
            algorithm(func)

            # Print some output if needed
            if verbose:
                print(func.state.current_best)

            # Reset the state of the problem
            func.reset()
    # Just to be sure, we close the logger explicitly
    if log_dir is not None:
        logger.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    #     dims = [2,3,4,6,8,10,15]
    #     fids = range(30,60)

    #     for fid in fids:
    #         for dim in dims:
    #             run_randomsearch(fid, dim, log_dir = ".", verbose = False)

    run_randomsearch(30, 2)
