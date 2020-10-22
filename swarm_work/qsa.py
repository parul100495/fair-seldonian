from sklearn.model_selection import train_test_split
from logistic_regression_functions import *
from scipy.optimize import minimize

import numpy as np
import pandas as pd

# Quasi-Seldonian algo
def QSA(X, Y, T, gHats, deltas, ineq):
    candidateData_len = 0.40
    candidateData_X, safetyData_X, candidateData_Y, safetyData_Y = train_test_split(
        X, Y, test_size = 1 - candidateData_len, shuffle = False)
    candidateData_T, safetyData_T = np.split(T, [int(candidateData_len * T.size), ])
    safetyData_X = safetyData_X.reset_index(drop=True)
    safetyData_T = pd.Series(safetyData_T.reset_index(drop=True))
    safetyData_Y = pd.Series(safetyData_Y.reset_index(drop=True))
    candidateSolution = getCandidateSolution(candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq, safetyData_X[1].count())
    print("candidate solution upperbound: ", gHat1(candidateSolution, candidateData_X, candidateData_Y, candidateData_T, deltas[0], ineq, False, None))
    if candidateSolution is not None:
        passedSafety = safetyTest(candidateSolution, safetyData_X, safetyData_Y, safetyData_T, gHats, deltas, ineq)
        return [candidateSolution, passedSafety]
    else:
        return [ None, False ]


# Run the safety test on a candidate solution. Returns true if the test is passed.
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
