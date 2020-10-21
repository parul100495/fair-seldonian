from numba import jit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from equation_parser import *

import math
import pandas as pd
import numpy as np


# simple logistic regression classifier
@jit(nopython=True)
def predict(theta, x):
    power_value = theta[5] + theta[0]*x[0] + theta[1]*x[1] + \
                  theta[2]*x[2] + theta[3]*x[3] + theta[4]*x[4]
    denominator = 1 + math.exp(-power_value)
    prob_value = 1 / denominator
    if prob_value > 0.5:
        return 1
    return 0


# Estimator of the primary objective - negative of the log loss
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
    rev_polish_notation = "TP(0) TP(1) - abs 0.2 TP(1) * -"
    r = construct_expr_tree(rev_polish_notation)
    _, u = eval_expr_tree_conf_interval(r, pd.Series(Y), pd.Series(predicted_Y), pd.Series(T), delta,
                                              ineq, predict_bound, d2)
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
        return np.array([theta1[0], theta1[1], theta1[2], theta1[3], theta1[4], theta0])
    except Exception as e:
        print("Exception in logRes:", e)
        return None
