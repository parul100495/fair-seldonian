import timeit
import numpy as np
import ray
import logging
logging.basicConfig(filename='main.py', level=logging.INFO)
ray.shutdown()
ray.init()
from synthetic_data import *
from qsa_base import *
from qsa_bound import *
from qsa_mod import *
from qsa_const import *
from qsa_opt import *
from logistic_regression_functions import *
# Folder where the experiment results will be saved
bin_path = 'exp/exp_base/bin/'


@ray.remote
def run_experiments(worker_id, nWorkers, ms, numM, numTrials, mTest, N):
    # Results of the Seldonian algorithm runs - base
    sbase_solutions_found = np.zeros((numTrials, numM))
    sbase_failures_g1 = np.zeros((numTrials, numM))
    sbase_upper_bound = np.zeros((numTrials, numM))
    sbase_fs = np.zeros((numTrials, numM))

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
                upper_bound = eval_ghat_base(theta, testX, testY, testT)
                if upper_bound > 0:
                    LS_failures_g1[trial, mIndex] = False
                else:
                    LS_failures_g1[trial, mIndex] = True
                LS_upper_bound[trial, mIndex] = upper_bound
                LS_fs[trial, mIndex] = -trueLogLoss
                print(
                    f"[(worker {worker_id}/{nWorkers}) simple_logistic   trial {trial + 1}/{numTrials}, m {m}]"
                    f"LS fHat over test data: {trueLogLoss:.10f}, upper bound: {upper_bound:.10f}")
            else:
                raise Exception("Dirty data. Raising exception.")

            # Run QSA Base
            (result, passedSafetyTest) = QSA_Base(trainX, trainY, trainT)
            if passedSafetyTest:
                sbase_solutions_found[trial, mIndex] = 1
                trueLogLoss = -fHat(result, testX, testY)
                upper_bound = eval_ghat_base(result, testX, testY, testT)
                if upper_bound > 0:
                    sbase_failures_g1[trial, mIndex] = False
                else:
                    sbase_failures_g1[trial, mIndex] = True
                sbase_upper_bound[trial, mIndex] = upper_bound
                sbase_fs[trial, mIndex] = -trueLogLoss
                print(
                    f"[(worker {worker_id}/{nWorkers}) SBase trial {trial + 1}/{numTrials}, m {m}]"
                    f"Solution found: [{result[0]:.10f}, {result[1]:.10f}]"
                    f"\tfHat over test data: {trueLogLoss:.10f}, upper bound: {upper_bound:.10f}")
            else:
                sbase_solutions_found[trial, mIndex], sbase_failures_g1[trial, mIndex], sbase_upper_bound[trial, mIndex] = 0, 0, 0
                sbase_fs[trial, mIndex] = None
                print(
                    f"[(worker {worker_id}/{nWorkers}) SBase trial {trial + 1}/{numTrials}, m {m}] No solution found")

    np.savez(outputFile,
            ms = ms,

            sbase_solutions_found = sbase_solutions_found,
            sbase_fs = sbase_fs,
            sbase_failures_g1 = sbase_failures_g1,
            sbase_upper_bound = sbase_upper_bound,

            LS_solutions_found = LS_solutions_found,
            LS_fs = LS_fs,
            LS_failures_g1 = LS_failures_g1,
            LS_upper_bound = LS_upper_bound)
    print(f"Saved the file {outputFile}")


if __name__ == "__main__":
    print("Assuming the default: 30")
    nWorkers = 30
    print(f"Running experiments on {nWorkers} threads")
    N = 40000
    ms = np.linspace(0.2, 1, num=12)  # 30 fractions
    numM = len(ms)
    numTrials = 1  # 24 * 5 = 120 samples per fraction
    mTest = 0.2  # about 0.2 * 10000 test samples = fraction of total data

    # Start 'nWorkers' threads in parallel, each one running 'numTrials' trials. Each thread saves its results to a file
    tic = timeit.default_timer()
    _ = ray.get(
        [run_experiments.remote(worker_id, nWorkers, ms, numM, numTrials, mTest, N) for worker_id in
          range(1, nWorkers + 1)])
    toc = timeit.default_timer()
    time_parallel = toc - tic  # Elapsed time in seconds
    print(f"Time elapsed: {time_parallel}")
    ray.shutdown()
