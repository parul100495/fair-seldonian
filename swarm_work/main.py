import timeit
import numpy as np
import ray
import logging
logging.basicConfig(filename='main.py', level=logging.INFO)
ray.shutdown()
ray.init()
from synthetic_data import *
from qsa import *
from logistic_regression_functions import *
# Folder where the experiment results will be saved
bin_path = 'experiment_results/bin/'


@ray.remote
def run_experiments(worker_id, nWorkers, ms, numM, numTrials, mTest, N, ineq):
    # Results of the Seldonian algorithm runs
    seldonian_solutions_found = np.zeros((numTrials, numM))
    seldonian_failures_g1 = np.zeros((numTrials, numM))
    seldonian_upper_bound = np.zeros((numTrials, numM))
    seldonian_fs = np.zeros((numTrials, numM))

    # Results of the logistic regression runs
    LS_solutions_found = np.zeros((numTrials, numM))
    LS_failures_g1 = np.zeros((numTrials, numM))
    LS_upper_bound = np.zeros((numTrials, numM))
    LS_fs = np.zeros((numTrials, numM))

    # Prepares file where experiment results will be saved
    experiment_number = worker_id
    outputFile = bin_path + 'results%d.npz' % experiment_number
    print("Writing output to", outputFile)

    # Generate the data used to evaluate the primary objective and failure rates
    for trial in range(numTrials):
        for (mIndex, m) in enumerate(ms):
            try:
                # Generate the training data, D
                base_seed = (experiment_number * numTrials) + 1
                random_state = base_seed + trial
                # Test data
                testX, testY, testT = get_data(mTest * N, 5, 0.5, 0.5, 0.7, trial + mIndex + experiment_number - 1)
                # Train data
                trainX, trainY, trainT = get_data(m * N, 5, 0.5, 0.5, 0.7, random_state)

                # Run the logistic regression algorithm
                theta = simple_logistic(trainX, trainY)  # Run least squares linear regression
                if theta is not None:
                    LS_solutions_found[trial, mIndex] = 1
                    trueLogLoss = -fHat(theta, testX, testY)
                    upper_bound = gHat1(theta, testX, testY, testT, deltas[0], ineq, False, None)
                    LS_failures_g1[trial, mIndex] = eval_ghat(theta, testX, testY, testT, deltas[0], ineq)  # Check if the first behavioral constraint was violated
                    LS_upper_bound[trial, mIndex] = upper_bound
                    LS_fs[trial, mIndex] = -trueLogLoss  # Store the "true" negative log loss

                    print(
                        f"[(worker {worker_id}/{nWorkers}) simple_logistic   trial {trial + 1}/{numTrials}, m {m}]"
                        f"LS fHat over test data: {trueLogLoss:.10f}, upper bound: {upper_bound:.10f}")
                else:
                    LS_solutions_found[trial, mIndex] = 0

                    LS_failures_g1[trial, mIndex] = 0
                    LS_upper_bound[trial, mIndex] = 0
                    LS_fs[trial, mIndex] = None
                    print(
                        f"[(worker {worker_id}/{nWorkers}) Dirty dataset  trial {trial + 1}/{numTrials}, m {m}]"
                         "No LS solution found. Skipped!")

                    seldonian_solutions_found[trial, mIndex] = 0

                    seldonian_failures_g1[trial, mIndex] = 0
                    seldonian_upper_bound[trial, mIndex] = 0
                    seldonian_fs[trial, mIndex] = None
                    print(
                        f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial + 1}/{numTrials}, m {m}]"
                        "No solution found")
                    continue

                # Run the Quasi-Seldonian algorithm
                (result, passedSafetyTest) = QSA(trainX, trainY, trainT, gHats, deltas, ineq)
                if passedSafetyTest:
                    seldonian_solutions_found[trial, mIndex] = 1
                    trueLogLoss = -fHat(result, testX, testY)
                    upper_bound = gHat1(result, testX, testY, testT, deltas[0], ineq, False, None)

                    seldonian_failures_g1[trial, mIndex] = eval_ghat(result, testX, testY, testT, deltas[0], ineq)
                    seldonian_upper_bound[trial, mIndex] = upper_bound
                    seldonian_fs[trial, mIndex] = -trueLogLoss
                    print(
                        f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial + 1}/{numTrials}, m {m}]"
                        f"Solution found: [{result[0]:.10f}, {result[1]:.10f}]"
                        f"\tfHat over test data: {trueLogLoss:.10f}, upper bound: {upper_bound:.10f}")
                else:
                    seldonian_solutions_found[trial, mIndex] = 0

                    seldonian_failures_g1[trial, mIndex] = 0
                    seldonian_fs[trial, mIndex] = None
                    seldonian_upper_bound[trial, mIndex] = 0
                    print(
                        f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial + 1}/{numTrials}, m {m}] No solution found")

            except Exception as e:
                logging.exception(e)

    np.savez(outputFile,
               ms = ms,
               seldonian_solutions_found = seldonian_solutions_found,
               seldonian_fs = seldonian_fs,
               seldonian_failures_g1 = seldonian_failures_g1,
               seldonian_upper_bound = seldonian_upper_bound,
               LS_solutions_found = LS_solutions_found,
               LS_fs = LS_fs,
               LS_failures_g1 = LS_failures_g1,
               LS_upper_bound = LS_upper_bound)
    print(f"Saved the file {outputFile}")


if __name__ == "__main__":

    # Create the behavioral constraints: each is a gHat function and a confidence level delta
    gHats = [gHat1]
    deltas = [0.05]

    # get data
    # _, _, _, All = get_data(1000000000, 5, 0.5, 0.5, 0.7)
    print("Assuming the default: 16")
    nWorkers = 2
    print(f"Running experiments on {nWorkers} threads")
    N = 1000
    ms = np.logspace(-1, 0, num=2)  # 6 fractions
    numM = len(ms)
    numTrials = 2  # 4 * 8 = 32 samples per fraction
    mTest = 0.2  # about 0.2 * 1000000000 test samples = fraction of total data

    # Start 'nWorkers' threads in parallel, each one running 'numTrials' trials. Each thread saves its results to a file
    tic = timeit.default_timer()
    _ = ray.get(
        [run_experiments.remote(worker_id, nWorkers, ms, numM, numTrials, mTest, N, Inequality.HOEFFDING_INEQUALITY) for worker_id in
          range(1, nWorkers + 1)])
    toc = timeit.default_timer()
    time_parallel = toc - tic  # Elapsed time in seconds
    print(f"Time elapsed: {time_parallel}")
    ray.shutdown()
