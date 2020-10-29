import timeit
import numpy as np
import ray
import logging
logging.basicConfig(filename='main.py', level=logging.INFO)
ray.shutdown()
ray.init()
import sys
from synthetic_data import *
from qsa_base import *
from qsa_bound import *
from qsa_mod import *
from qsa_const import *
from qsa_opt import *
from logistic_regression_functions import *
# Folder where the experiment results will be saved
bin_path = 'exp/exp_{}/bin/'


@ray.remote
def run_experiments(worker_id, nWorkers, ms, numM, numTrials, mTest, N, seldonian_type):
    if seldonian_type == 'base':
        # Results of the Seldonian algorithm runs - base
        sbase_solutions_found = np.zeros((numTrials, numM))
        sbase_failures_g1 = np.zeros((numTrials, numM))
        sbase_upper_bound = np.zeros((numTrials, numM))
        sbase_fs = np.zeros((numTrials, numM))

    if seldonian_type == 'mod':
        # Results of the Seldonian algorithm runs - modified h
        smod_solutions_found = np.zeros((numTrials, numM))
        smod_failures_g1 = np.zeros((numTrials, numM))
        smod_upper_bound = np.zeros((numTrials, numM))
        smod_fs = np.zeros((numTrials, numM))

    if seldonian_type == 'const':
        # Results of the Seldonian algorithm runs - constant change
        sconst_solutions_found = np.zeros( (numTrials, numM))
        sconst_failures_g1 = np.zeros((numTrials, numM))
        sconst_upper_bound = np.zeros((numTrials, numM))
        sconst_fs = np.zeros((numTrials, numM))

    if seldonian_type == 'bound':
        # Results of the Seldonian algorithm runs - union bound
        sbound_solutions_found = np.zeros((numTrials, numM))
        sbound_failures_g1 = np.zeros((numTrials, numM))
        sbound_upper_bound = np.zeros((numTrials, numM))
        sbound_fs = np.zeros((numTrials, numM))

    if seldonian_type == 'opt':
        # Results of the Seldonian algorithm runs - optimized
        sopt_solutions_found = np.zeros((numTrials, numM))
        sopt_failures_g1 = np.zeros((numTrials, numM))
        sopt_upper_bound = np.zeros((numTrials, numM))
        sopt_fs = np.zeros((numTrials, numM))

    # Results of the logistic regression runs
    LS_solutions_found = np.zeros((numTrials, numM))
    LS_failures_g1 = np.zeros((numTrials, numM))
    LS_upper_bound = np.zeros((numTrials, numM))
    LS_fs = np.zeros((numTrials, numM))

    # Prepares file where experiment results will be saved
    experiment_number = worker_id
    outputFile = bin_path.format(seldonian_type) + 'results%d.npz' % experiment_number
    print("Writing output to", outputFile)

    # Generate the data used to evaluate the primary objective and failure rates
    for trial in range(numTrials):
        for (mIndex, m) in enumerate(ms):
            # Generate the training data, D
            base_seed = (experiment_number * numTrials) + 1
            random_state = base_seed + trial
            # Test data
            testX, testY, testT = get_data(mTest * N, 5, 0.5, 0.4, 0.6, trial + mIndex + experiment_number - 1)
            # Train data
            trainX, trainY, trainT = get_data(m * N, 5, 0.5, 0.4, 0.6, random_state)

            # Run the logistic regression algorithm
            theta = simple_logistic(trainX, trainY)  # Run least squares linear regression
            if theta is not None:
                LS_solutions_found[trial, mIndex] = 1
                trueLogLoss = -fHat(theta, testX, testY)
                upper_bound = eval_ghat_base(theta, testX, testY, testT)
                if upper_bound > 0:
                    LS_failures_g1[trial, mIndex] = 1
                else:
                    LS_failures_g1[trial, mIndex] = 0
                LS_upper_bound[trial, mIndex] = upper_bound
                LS_fs[trial, mIndex] = -trueLogLoss
                print(
                    f"[(worker {worker_id}/{nWorkers}) simple_logistic   trial {trial + 1}/{numTrials}, m {m}]"
                    f"LS fHat over test data: {trueLogLoss:.10f}, upper bound: {upper_bound:.10f}")
            else:
                raise Exception("Dirty data. Raising exception.")

            if seldonian_type == 'base':
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

            if seldonian_type == 'mod':
                # Run QSA hoeffding modified
                (result, passedSafetyTest) = QSA_Mod(trainX, trainY, trainT)
                if passedSafetyTest:
                    smod_solutions_found[trial, mIndex] = 1
                    trueLogLoss = -fHat(result, testX, testY)
                    upper_bound = eval_ghat_mod(result, testX, testY, testT)
                    if upper_bound > 0:
                        smod_failures_g1[trial, mIndex] = 1
                    else:
                        smod_failures_g1[trial, mIndex] = 0
                    smod_upper_bound[trial, mIndex] = upper_bound
                    smod_fs[trial, mIndex] = -trueLogLoss
                    print(
                        f"[(worker {worker_id}/{nWorkers}) SMod trial {trial + 1}/{numTrials}, m {m}]"
                        f"Solution found: [{result [ 0 ]:.10f}, {result [ 1 ]:.10f}]"
                        f"\tfHat over test data: {trueLogLoss:.10f}, upper bound: {upper_bound:.10f}")
                else:
                    smod_solutions_found[trial, mIndex], smod_failures_g1[trial, mIndex], smod_upper_bound[trial, mIndex] = 0, 0, 0
                    smod_fs[trial, mIndex] = None
                    print(
                        f"[(worker {worker_id}/{nWorkers}) SMod trial {trial + 1}/{numTrials}, m {m}] No solution found")

            if seldonian_type == 'const':
                # Run QSA constant
                (result, passedSafetyTest) = QSA_Const(trainX, trainY, trainT)
                if passedSafetyTest:
                    sconst_solutions_found[trial, mIndex] = 1
                    trueLogLoss = -fHat(result, testX, testY)
                    upper_bound = eval_ghat_const(result, testX, testY, testT)
                    if upper_bound > 0:
                        sconst_failures_g1[trial, mIndex] = 1
                    else:
                        sconst_failures_g1[trial, mIndex] = 0
                    sconst_upper_bound[trial, mIndex] = upper_bound
                    sconst_fs[trial, mIndex] = -trueLogLoss
                    print(
                        f"[(worker {worker_id}/{nWorkers}) SConst trial {trial + 1}/{numTrials}, m {m}]"
                        f"Solution found: [{result [ 0 ]:.10f}, {result [ 1 ]:.10f}]"
                        f"\tfHat over test data: {trueLogLoss:.10f}, upper bound: {upper_bound:.10f}")
                else:
                    sconst_solutions_found[trial, mIndex], sconst_failures_g1[trial, mIndex], sconst_upper_bound[trial, mIndex] = 0, 0, 0
                    sconst_fs[trial, mIndex] = None
                    print(
                        f"[(worker {worker_id}/{nWorkers}) SConst trial {trial + 1}/{numTrials}, m {m}] No solution found")

            if seldonian_type == 'bound':
                # Run QSA bound
                (result, passedSafetyTest) = QSA_Bound(trainX, trainY, trainT)
                if passedSafetyTest:
                    sbound_solutions_found[trial, mIndex] = 1
                    trueLogLoss = -fHat(result, testX, testY)
                    upper_bound = eval_ghat_bound(result, testX, testY, testT)
                    if upper_bound > 0:
                        sbound_failures_g1[trial, mIndex] = 1
                    else:
                        sbound_failures_g1[trial, mIndex] = 0
                    sbound_upper_bound[trial, mIndex] = upper_bound
                    sbound_fs[trial, mIndex] = -trueLogLoss
                    print(
                        f"[(worker {worker_id}/{nWorkers}) SBound trial {trial + 1}/{numTrials}, m {m}]"
                        f"Solution found: [{result [ 0 ]:.10f}, {result [ 1 ]:.10f}]"
                        f"\tfHat over test data: {trueLogLoss:.10f}, upper bound: {upper_bound:.10f}")
                else:
                    sbound_solutions_found[trial, mIndex], sbound_failures_g1[trial, mIndex], sbound_upper_bound[trial, mIndex] = 0, 0, 0
                    sbound_fs[trial, mIndex] = None
                    print(
                        f"[(worker {worker_id}/{nWorkers}) SBound trial {trial + 1}/{numTrials}, m {m}] No solution found")

            if seldonian_type == 'opt':
                # Run QSA Optimized
                (result, passedSafetyTest) = QSA_Opt(trainX, trainY, trainT)
                if passedSafetyTest:
                    sopt_solutions_found[trial, mIndex] = 1
                    trueLogLoss = -fHat(result, testX, testY)
                    upper_bound = eval_ghat_opt(result, testX, testY, testT)
                    if upper_bound > 0:
                        sopt_failures_g1[trial, mIndex] = 1
                    else:
                        sopt_failures_g1[trial, mIndex] = 0
                    sopt_upper_bound[trial, mIndex] = upper_bound
                    sopt_fs[trial, mIndex] = -trueLogLoss
                    print(
                        f"[(worker {worker_id}/{nWorkers}) SOpt trial {trial + 1}/{numTrials}, m {m}]"
                        f"Solution found: [{result [ 0 ]:.10f}, {result [ 1 ]:.10f}]"
                        f"\tfHat over test data: {trueLogLoss:.10f}, upper bound: {upper_bound:.10f}" )
                else:
                    sopt_solutions_found[trial, mIndex], sopt_failures_g1[trial, mIndex], sopt_upper_bound[trial, mIndex] = 0, 0, 0
                    sopt_fs[trial, mIndex] = None
                    print(
                        f"[(worker {worker_id}/{nWorkers}) SOpt trial {trial + 1}/{numTrials}, m {m}] No solution found" )

    if seldonian_type == 'base':
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

    if seldonian_type == 'opt':
        np.savez(outputFile,
            sopt_solutions_found = sopt_solutions_found,
            sopt_fs = sopt_fs,
            sopt_failures_g1 = sopt_failures_g1,
            sopt_upper_bound = sopt_upper_bound,
            LS_solutions_found = LS_solutions_found,
            LS_fs = LS_fs,
            LS_failures_g1 = LS_failures_g1,
            LS_upper_bound = LS_upper_bound)

    if seldonian_type == 'mod':
        np.savez(outputFile,
            smod_solutions_found = smod_solutions_found,
            smod_fs = smod_fs,
            smod_failures_g1 = smod_failures_g1,
            smod_upper_bound = smod_upper_bound,
            LS_solutions_found = LS_solutions_found,
            LS_fs = LS_fs,
            LS_failures_g1 = LS_failures_g1,
            LS_upper_bound = LS_upper_bound)

    if seldonian_type == 'const':
        np.savez(outputFile,
            sconst_solutions_found = sconst_solutions_found,
            sconst_fs = sconst_fs,
            sconst_failures_g1 = sconst_failures_g1,
            sconst_upper_bound = sconst_upper_bound,
            LS_solutions_found = LS_solutions_found,
            LS_fs = LS_fs,
            LS_failures_g1 = LS_failures_g1,
            LS_upper_bound = LS_upper_bound)

    if seldonian_type == 'bound':
        np.savez(outputFile,
            sbound_solutions_found = sbound_solutions_found,
            sbound_fs = sbound_fs,
            sbound_failures_g1 = sbound_failures_g1,
            sbound_upper_bound = sbound_upper_bound,

            LS_solutions_found = LS_solutions_found,
            LS_fs = LS_fs,
            LS_failures_g1 = LS_failures_g1,
            LS_upper_bound = LS_upper_bound)
    print(f"Saved the file {outputFile}")


if __name__ == "__main__":
    print("Assuming the default: 50")
    nWorkers = 50
    print(f"Running experiments on {nWorkers} threads")
    N = 100000
    ms = np.logspace(-3, 1, num=20)  # 30 fractions
    print("N {}, frac array: {}".format(N, ms))
    print("Running for: ", sys.argv[1])
    numM = len(ms)
    numTrials = 1  # 24 * 5 = 120 samples per fraction
    mTest = 0.2  # about 0.2 * 10000 test samples = fraction of total data
    print("Number of trials: ", numTrials)
    # Start 'nWorkers' threads in parallel, each one running 'numTrials' trials. Each thread saves its results to a file
    tic = timeit.default_timer()
    _ = ray.get(
        [run_experiments.remote(worker_id, nWorkers, ms, numM, numTrials, mTest, N, sys.argv[1]) for worker_id in
          range(1, nWorkers + 1)])
    toc = timeit.default_timer()
    time_parallel = toc - tic  # Elapsed time in seconds
    print(f"Time elapsed: {time_parallel}")
    ray.shutdown()
