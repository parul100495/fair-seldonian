from helper import *  # Basic helper functions
import timeit  # To time the execution of ours experiments
from numba import jit  # Just-in-Time (JIT) compiler to accelerate Python code
import math
import pandas as pd
import numpy as np
import ray
import logging
logging.basicConfig(filename='main_adult.py',level=logging.INFO)
ray.shutdown()
ray.init()
from equation_parser import *
from test_data import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import shap
#shap.initjs()
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Folder where the experiment results will be saved
bin_path = 'experiment_results/bin/'

def get_adult_data():
    X, Y = shap.datasets.adult()
    sensitive_attribute = 'Sex'

    T = X[sensitive_attribute]
    X = pd.get_dummies(X)

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    le = LabelEncoder()
    Y = le.fit_transform(Y)

    X = X_scaled.reset_index(drop=True)
    T = T.reset_index(drop=True)
    X = pd.DataFrame(X)
    T = pd.Series(T)
    Y = pd.Series(Y)
    All = pd.concat([X, Y, T], axis = 1)
    return X, Y, T, All

# Generate numPoints data points
def generateData(frac, All, random_state, mTest):
    all_train, all_test, y_train, y_test = train_test_split(All, All.iloc[:, -1], test_size=mTest, random_state=42)
    # test dataset
    test_tmp = all_test.iloc[:, :-1]
    T_test = all_test.iloc[:, -1]
    X_test = test_tmp.iloc[:, :-1]
    Y_test = test_tmp.iloc[:, -1]

    # train dataset
    subsampling = all_train.sample(frac=frac, random_state=random_state)
    subsampling = subsampling.reset_index()
    subsampling = subsampling.drop(columns=['index'])
    tmp = subsampling.iloc[:, :-1]
    T = subsampling.iloc[:, -1]
    X = tmp.iloc[:, :-1]
    Y = tmp.iloc[:, -1]
    return np.array(X_test), np.array(Y_test), np.array(T_test), np.array(X), np.array(Y), np.array(T)


# simple logistic regression classifier
@jit(nopython=True)
def predict(theta, x):
    # power_value = theta[12] + theta[0]*x[0] + theta[1]*x[1] + theta[2]*x[2] +\
    #               theta[3]*x[3] + theta[4]*x[4] + theta[5]*x[5] + theta[6]*x[6] +\
    #               theta[7]*x[7] + theta[8]*x[8] + theta[9]*x[9] + theta[10]*x[10] +\
    #               theta[11]*x[11]
    power_value = theta[5] + theta[0]*x[0] + theta[1]*x[1] + theta[2]*x[2] + theta[3]*x[3] + theta[4]*x[4]
    denominator = 1 + math.exp(-power_value)
    prob_value = 1 / denominator
    if prob_value > 0.5:
        return 1
    return 0


# Estimator of the primary objective - negative of the log loss
# @jit(nopython=True)
def fHat(theta, X, Y):
    n = X.shape[0]
    predicted_Y = np.empty((n,))
    for i in range(n):
        predicted_Y[i] = predict(theta, X[i])
    res = log_loss(Y, predicted_Y)
    return -res


# Fairness constraint - True positive rate difference less than 0.1
def gHat1(theta, X, Y, T, delta, ineq, predict_bound, d2):
    n = X.shape[0]
    predicted_Y = np.empty((n,))
    for i in range(n):
        predicted_Y[i] = predict(theta, X[i])
    rev_polish_notation = "TP(0) TP(1) - abs 0.1 -"
    r = construct_expr_tree(rev_polish_notation)
    _, u = eval_expr_tree_conf_interval(r, pd.Series(Y), pd.Series(predicted_Y), pd.Series(T), delta,
                                              ineq, predict_bound, d2)
    # print(u)
    return u


def eval_ghat(theta, X, Y, T, delta, ineq):
    u = gHat1(theta, X, Y, T, delta, ineq, False, None)
    if u <= 0:
        return 0
    return 1


def simple_logistic(X, Y):
    try:
        reg = LogisticRegression(solver = 'lbfgs').fit(X, Y)
        theta0 = reg.intercept_[0]
        theta1 = reg.coef_[0]
        # return np.array([theta1[0], theta1[1], theta1[2], theta1[3], theta1[4], theta1[5],
        #                  theta1[6], theta1[7], theta1[8], theta1[9], theta1[10], theta1[11], theta0])
        return np.array([theta1[0], theta1[1], theta1[2], theta1[3], theta1[4], theta0])
    except Exception as e:
        print("Exception in logRes:", e)
        return None


# Quasi-Seldonian algo
def QSA(X, Y, T, gHats, deltas, ineq):
    candidateData_len = 0.40
    candidateData_X, safetyData_X, candidateData_Y, safetyData_Y = train_test_split (
        X, Y, test_size = 1 - candidateData_len, shuffle = False )
    candidateData_T, safetyData_T = np.split(T, [int(candidateData_len * T.size), ])

    candidateSolution = getCandidateSolution(candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq, safetyData_X.shape[0])
    print("candidate solution upperbound: ", gHat1(candidateSolution, candidateData_X, candidateData_Y, candidateData_T, deltas[0], ineq, False, None))
    if candidateSolution is not None:
        passedSafety = safetyTest(candidateSolution, safetyData_X, safetyData_Y, safetyData_T, gHats, deltas, ineq)
        return [candidateSolution, passedSafety]
    else:
        return [ None, False ]


# Run the safety test on a candidate solution. Returns true if the test is passed.
#   candidateSolution: the solution to test.
#   (safetyData_X, safetyData_Y): data set D2 to be used in the safety test.
#   (gHats, deltas): vectors containing the behavioral constraints and confidence levels.
def safetyTest(candidateSolution, safetyData_X, safetyData_Y, safetyData_T, gHats, deltas, ineq):
    for i in range(len(gHats)):
        g = gHats[i]
        delta = deltas[i]
        upperBound = g(candidateSolution, safetyData_X, safetyData_Y, safetyData_T, delta, ineq, False, None)
        print("safety test upper bound: ", upperBound)
        if upperBound > 0.0:
            return False
    return True


# The objective function maximized by getCandidateSolution.
#     thetaToEvaluate: the candidate solution to evaluate.
#     (candidateData_X, candidateData_Y): the data set D1 used to evaluated the solution.
#     (gHats, deltas): vectors containing the behavioral constraints and confidence levels.
#     safetyDataSize: |D2|, used when computing the conservative upper bound on each behavioral constraint.
def candidateObjective(thetaToEvaluate, candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq, safety_size):
    result = fHat(thetaToEvaluate, candidateData_X, candidateData_Y)
    predictSafetyTest = True
    for i in range(len(gHats)):
        g = gHats[i]
        delta = deltas[i]
        upperBound = g(thetaToEvaluate, candidateData_X, candidateData_Y, candidateData_T, delta, ineq, True, safety_size)
        if upperBound > 0.0:
            if predictSafetyTest:
                predictSafetyTest = False
                result = -1000.0
            result = result - upperBound
    return -result


# Use the provided data to get a candidate solution expected to pass the safety test.
#    (candidateData_X, candidateData_Y): data used to compute a candidate solution.
#    (gHats, deltas): vectors containing the behavioral constraints and confidence levels.
#    safetyDataSize: |D2|, used when computing the conservative upper bound on each behavioral constraint.
def getCandidateSolution(candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq, safety_size):
    minimizer_method = 'Powell'
    minimizer_options = {'disp': False}
    initialSolution = simple_logistic(candidateData_X, candidateData_Y)
    print("LS upperbound: ", gHat1(initialSolution, candidateData_X, candidateData_Y, candidateData_T, deltas[0], ineq, False, None))
    if initialSolution is not None:
        res = minimize(candidateObjective, x0 = initialSolution, method = minimizer_method,
                         options = minimizer_options,
                         args = (candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq, safety_size))
        return res.x
    else:
        return None


@ray.remote
def run_experiments(worker_id, nWorkers, ms, numM, numTrials, mTest, All, ineq):
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
                (testX, testY, testT, trainX, trainY, trainT) = generateData(m, All, random_state, mTest)

                # Run the logistic regression algorithm
                theta = simple_logistic(trainX, trainY)  # Run least squares linear regression
                if theta is not None:
                    LS_solutions_found[trial, mIndex] = 1
                    trueLogLoss = -fHat(theta, testX, testY)
                    upper_bound = gHat1(theta, testX, testY, testT, deltas[0], ineq, False, None)
                    LS_failures_g1[trial, mIndex] = eval_ghat(theta, testX, testY, testT, deltas[0], ineq)  # Check if the first behavioral constraint was violated
                    LS_upper_bound[trial, mIndex] = upper_bound
                    LS_fs[trial, mIndex] = -trueLogLoss  # Store the "true" negative mean-squared error

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
    #_, _, _, All = get_adult_data()

    _, _, _, All = get_data(30000, 5, 0.5, 0.55, 0.7)
    print("Assuming the default: 16")
    nWorkers = 2
    print(f"Running experiments on {nWorkers} threads")

    ms = np.logspace(-1.4, 0, num=2)
    numM = len(ms)
    numTrials = 2  # 65 * 16 = 1040 samples per fraction
    mTest = 0.2  # about 6500 test samples = fraction of total data

    # Start 'nWorkers' threads in parallel, each one running 'numTrials' trials. Each thread saves its results to a file
    tic = timeit.default_timer()
    _ = ray.get(
        [run_experiments.remote(worker_id, nWorkers, ms, numM, numTrials, mTest, All, Inequality.HOEFFDING_INEQUALITY) for worker_id in
          range(1, nWorkers + 1)])
    toc = timeit.default_timer()
    time_parallel = toc - tic  # Elapsed time in seconds
    print(f"Time elapsed: {time_parallel}")
    ray.shutdown()
