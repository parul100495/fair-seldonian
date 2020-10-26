from sklearn.model_selection import train_test_split
from logistic_regression_functions import *
from scipy.optimize import minimize

import numpy as np
import pandas as pd

# QSA const
def QSA_Const(X, Y, T):
    candidateData_len = 0.40
    candidateData_X, safetyData_X, candidateData_Y, safetyData_Y = train_test_split(
        X, Y, test_size = 1 - candidateData_len, shuffle = False)
    candidateData_T, safetyData_T = np.split(T, [int(candidateData_len * T.size), ])
    safetyData_X = safetyData_X.reset_index(drop=True)
    safetyData_T = pd.Series(safetyData_T.reset_index(drop=True))
    safetyData_Y = pd.Series(safetyData_Y.reset_index(drop=True))

    candidateSolution = getCandidateSolution(candidateData_X, candidateData_Y, candidateData_T, safetyData_X[1].count())
    print("Actual candidate sol upperbound: ", eval_ghat_const(candidateSolution, candidateData_X, candidateData_Y, candidateData_T))
    if candidateSolution is not None:
        passedSafety = safetyTest(candidateSolution, safetyData_X, safetyData_Y, safetyData_T)
        return [candidateSolution, passedSafety]
    else:
        return [None, False]


def safetyTest(candidateSolution, safetyData_X, safetyData_Y, safetyData_T):
    upperBound = eval_ghat_const(candidateSolution, safetyData_X, safetyData_Y, safetyData_T)
    print("Safety test upperbound: ", upperBound)
    if upperBound > 0.0:
        return False
    return True


def candidateObjective(thetaToEvaluate, candidateData_X, candidateData_Y, candidateData_T, safety_size):
    result = fHat(thetaToEvaluate, candidateData_X, candidateData_Y)
    upperBound = ghat_const(thetaToEvaluate, candidateData_X, candidateData_Y, candidateData_T, True, safety_size)
    if upperBound > 0.0:
        result = -10000.0 - upperBound
    return -result


def getCandidateSolution(candidateData_X, candidateData_Y, candidateData_T, safety_size):
    initialSolution = simple_logistic(candidateData_X, candidateData_Y)
    print("Initial LS upperbound: ", eval_ghat_const(initialSolution, candidateData_X, candidateData_Y, candidateData_T))
    if initialSolution is not None:
        res = minimize(candidateObjective, x0 = initialSolution, method = 'Powell',
                     options = {'disp': False, 'maxiter': 10000},
                     args = (candidateData_X, candidateData_Y, candidateData_T, safety_size))
        return res.x
    else:
        return None
