from sklearn.linear_model import LogisticRegression
from equation_parser import *
from inequalities import *
from equation_parser_extension import *
import numpy as np
import torch

delta = 0.05
ineq = Inequality.HOEFFDING_INEQUALITY
rev_polish_notation = "TP(1) TP(0) - abs 0.2 TP(1) * -"
candidate_ratio = 0.40


def predict(theta, theta1, X):
    # returns tensor
    # \frac{1}{1 + e^-(X.theta + theta1)}
    if theta1 is None or theta is None:
        return torch.ones(len(X))
    return torch.pow(
                torch.add(
                    torch.exp(
                        torch.mul(-1,
                                  torch.add(torch.matmul(torch.tensor(X), theta), theta1)
                                  )
                                )
                    , 1),
                -1)


def fHat(theta, theta1, X, Y):
    # -ve log loss
    pred = predict(theta, theta1, X)
    predicted_Y = torch.stack([torch.sub(1, pred), pred], dim = 1)
    loss = torch.nn.CrossEntropyLoss()
    return -loss(predicted_Y, torch.tensor(Y).long())


def simple_logistic(X, Y):
    # return tensor
    try:
        reg = LogisticRegression(solver = 'lbfgs').fit(X, Y)
        theta0 = reg.intercept_[0]
        theta1 = reg.coef_[0]
        return torch.tensor(np.array([theta1[0], theta1[1], theta1[2], theta1[3], theta1[4]]),
                            requires_grad=True),\
               torch.tensor(np.array([theta0]),
                            requires_grad=True)
    except Exception as e:
        print("Exception in logRes:", e)
        return None


def eval_ghat(theta, theta1, X, Y, T, seldonian_type):
    if seldonian_type == "base":
        return eval_ghat_base(theta, theta1, X, Y, T, False)
    elif seldonian_type == "mod":
        return eval_ghat_base(theta, theta1, X, Y, T, True)
    elif seldonian_type == "bound":
        return eval_ghat_extend(theta, theta1, X, Y, T, True, False, False)
    elif seldonian_type == "const":
        return eval_ghat_extend(theta, theta1, X, Y, T, False, True, False)
    elif seldonian_type == "opt":
        return eval_ghat_extend(theta, theta1, X, Y, T, True, True, True)


def ghat(theta, theta1, X, Y, T, candidate_ratio, seldonian_type):
    if seldonian_type == "base":
        return ghat_base(theta, theta1, X, Y, T, True, candidate_ratio, False)
    elif seldonian_type == "mod":
        return ghat_base(theta, theta1, X, Y, T, True, candidate_ratio, True)
    elif seldonian_type == "bound":
        return ghat_extend(theta, theta1, X, Y, T, True, candidate_ratio, True, False, False)
    elif seldonian_type == "const":
        return ghat_extend(theta, theta1, X, Y, T, True, candidate_ratio, False, True, False)
    elif seldonian_type == "opt":
        return ghat_extend(theta, theta1, X, Y, T, True, candidate_ratio, True, True, True)

def ghat_base(theta, theta1, X, Y, T, predict_bound, candidate_ratio, modified_h):
    pred = predict(theta, theta1, X)
    r = construct_expr_tree_base(rev_polish_notation)
    cand_safe_ratio = None
    if candidate_ratio:
        cand_safe_ratio = (1-candidate_ratio)/candidate_ratio
    _, u = eval_expr_tree_conf_interval_base(t_node=r, Y=Y, predicted_Y=pred, T=T,
                                             delta=delta, inequality=ineq,
                                             candidate_safety_ratio=cand_safe_ratio,
                                             predict_bound=predict_bound, modified_h=modified_h)
    return u


def eval_ghat_base(theta, theta1, X, Y, T, modified_h):
    return ghat_base(theta, theta1, X, Y, T, False, None, modified_h)


def ghat_extend(theta, theta1, X, Y, T, predict_bound, candidate_ratio,
                check_bound, check_const, modified_h):
    pred = predict(theta, theta1, X)
    r = construct_expr_tree(rev_polish_notation, delta,
                            check_bound=check_bound, check_constant=check_const)
    cand_safe_ratio = None
    if candidate_ratio:
        cand_safe_ratio = (1 - candidate_ratio) / candidate_ratio
    _, u = eval_expr_tree_conf_interval(t_node=r, Y=Y, predicted_Y=pred, T=T,
                                        inequality=ineq, candidate_safety_ratio=cand_safe_ratio,
                                        predict_bound=predict_bound, modified_h=modified_h)
    return u


def eval_ghat_extend(theta, theta1, X, Y, T, check_bound, check_const, modified_h):
    return ghat_extend(theta, theta1, X, Y, T, False, None, check_bound, check_const, modified_h)
