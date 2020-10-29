from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from equation_parser import *
from equation_parser_extension import *

import math
import pandas as pd
import numpy as np


delta = 0.05
ineq = Inequality.HOEFFDING_INEQUALITY
rev_polish_notation = "TP(1) TP(0) - 0.3 TP(0) * -"

def predict(theta, x):
    power_value = theta[3] + theta[0]*x.iloc[0] + theta[1]*x.iloc[1] + \
                  theta[2]*x.iloc[2] #  + theta[3]*x.iloc[3] + theta[4]*x.iloc[4]
    try:
    	prob_value = 1 / (1 + math.exp(-power_value))
    except OverflowError:
    	prob_value = 0
    if prob_value > 0.5:
        return 1
    return 0


def fHat(theta, X, Y):
    n = len(Y)
    predicted_Y = np.empty((n,))
    for i in range(n):
        predicted_Y[i] = predict(theta, X.iloc[i])
    res = log_loss(Y, predicted_Y)
    return -res


def simple_logistic(X, Y):
    try:
        reg = LogisticRegression(solver = 'lbfgs').fit(X, Y)
        theta0 = reg.intercept_[0]
        theta1 = reg.coef_[0]
        return np.array([theta1[0], theta1[1], theta1[2], #  theta1[3], theta1[4],
                        theta0])
    except Exception as e:
        print("Exception in logRes:", e)
        return None


def ghat_base(theta, X, Y, T, predict_bound, d2):
    n = len(Y)
    predicted_Y = np.empty((n,))
    for i in range(n):
        predicted_Y[i] = predict(theta, X.iloc[i])
    r = construct_expr_tree_base(rev_polish_notation)
    predicted_Y = pd.Series(predicted_Y)
    l, u = eval_expr_tree_conf_interval_base(t_node=r, Y=Y, predicted_Y=predicted_Y, T=T, delta=delta,
                                        inequality=ineq, predict_bound=predict_bound, safety_size=d2,
                                        modified_h=False)
    return u


def eval_ghat_base(theta, X, Y, T):
    return ghat_base(theta, X, Y, T, False, None)


def ghat_mod(theta, X, Y, T, predict_bound, d2):
    n = X[1].count()
    predicted_Y = np.empty((n,))
    for i in range(n):
        predicted_Y[i] = predict(theta, X.iloc[i])
    r = construct_expr_tree_base(rev_polish_notation)
    predicted_Y = pd.Series(predicted_Y)
    l, u = eval_expr_tree_conf_interval_base(t_node=r, Y=Y, predicted_Y=predicted_Y, T=T, delta=delta,
                                        inequality=ineq, predict_bound=predict_bound, safety_size=d2,
                                        modified_h=True)
    return u


def eval_ghat_mod(theta, X, Y, T):
    return ghat_mod(theta, X, Y, T, False, None)


def ghat_bound(theta, X, Y, T, predict_bound, d2):
    n = X[1].count()
    predicted_Y = np.empty((n,))
    for i in range(n):
        predicted_Y[i] = predict(theta, X.iloc[i])
    r = construct_expr_tree(rev_polish_notation, delta, check_bound=True, check_constant=False)
    predicted_Y = pd.Series(predicted_Y)
    l, u = eval_expr_tree_conf_interval(t_node=r, Y=Y, predicted_Y=predicted_Y, T=T, inequality=ineq,
                                        predict_bound=predict_bound, safety_size=d2, modified_h=False)
    return u


def eval_ghat_bound(theta, X, Y, T):
    return ghat_bound(theta, X, Y, T, False, None)


def predict_ghat_const(theta, X, Y, T, predict_bound, d2):
    n = len(Y)
    predicted_Y = np.empty((n,))
    for i in range(n):
        predicted_Y[i] = predict(theta, X.iloc[i])
    r = construct_expr_tree(rev_polish_notation, delta, check_bound=False, check_constant=True)
    predicted_Y = pd.Series(predicted_Y)
    l, u = eval_expr_tree_conf_interval(t_node=r, Y=Y, predicted_Y=predicted_Y, T=T, inequality=ineq,
                                        predict_bound=predict_bound, safety_size=d2, modified_h=False)
    return u


def eval_ghat_const(theta, X, Y, T):
    return predict_ghat_const(theta, X, Y, T, False, None)

def ghat_opt(theta, X, Y, T, predict_bound, d2):
    n = X[1].count()
    predicted_Y = np.empty((n,))
    for i in range(n):
        predicted_Y[i] = predict(theta, X.iloc[i])
    r = construct_expr_tree(rev_polish_notation, delta, check_bound=True, check_constant=True)
    predicted_Y = pd.Series(predicted_Y)
    l, u = eval_expr_tree_conf_interval(t_node=r, Y=Y, predicted_Y=predicted_Y, T=T, inequality=ineq,
                                        predict_bound=predict_bound, safety_size=d2, modified_h=True)
    return u


def eval_ghat_opt(theta, X, Y, T):
    return ghat_opt(theta, X, Y, T, False, None)
