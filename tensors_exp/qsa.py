from sklearn.model_selection import train_test_split
from logistic_regression_functions import *
from scipy.optimize import minimize
import numpy as np
import torch
candidate_ratio = 0.40


# QSA
def QSA(X, Y, T, seldonian_type):
    cand_data_X, safe_data_X, cand_data_Y, safe_data_Y = train_test_split(X, Y,
                                                                          test_size = 1 - candidate_ratio,
                                                                          shuffle = False)
    cand_data_T, safe_data_T = np.split(T, [int(candidate_ratio * T.size),])

    theta, theta1 = get_cand_solution(cand_data_X, cand_data_Y, cand_data_T, candidate_ratio, seldonian_type)
    print("Actual cand sol upperbound: ", eval_ghat(theta, theta1,
                                                    cand_data_X, cand_data_Y, cand_data_T,
                                                    seldonian_type))
    passed_safety = safety_test(theta, theta1, safe_data_X, safe_data_Y, safe_data_T, seldonian_type)
    return [theta, theta1, passed_safety]


def safety_test(theta, theta1, safe_data_X, safe_data_Y, safe_data_T, seldonian_type):
    upper_bound = eval_ghat(theta, theta1, safe_data_X, safe_data_Y, safe_data_T, seldonian_type)
    print("Safety test upperbound: ", upper_bound)
    if upper_bound > 0.0:
        return False
    return True


def get_cand_solution(cand_data_X, cand_data_Y, cand_data_T, candidate_ratio, seldonian_type):
    init_sol, init_sol1 = simple_logistic(cand_data_X, cand_data_Y)
    print("Initial LS upperbound: ", eval_ghat(init_sol, init_sol1,
                                               cand_data_X, cand_data_Y, cand_data_T,
                                               seldonian_type))
    theta = init_sol.detach().numpy()
    theta1 = init_sol1.detach().numpy()
    init_theta = np.concatenate((theta, theta1))
    res = minimize(cand_obj, x0 = init_theta, method = 'Powell',
                     options = {'disp': False, 'maxiter': 10000},
                     args = (cand_data_X, cand_data_Y, cand_data_T, candidate_ratio, seldonian_type))
    # ndarray -> tensor of theta
    theta_numpy = res.x[:-1]
    theta1_numpy = res.x[-1]
    theta0 = torch.tensor(theta_numpy)
    theta1 = torch.tensor(np.array([theta1_numpy]))
    return theta0, theta1


def cand_obj(theta, cand_data_X, cand_data_Y, cand_data_T, candidate_ratio, seldonian_type):
    theta_numpy = theta[:-1]
    theta1_numpy = theta[-1]
    theta0 = torch.tensor(theta_numpy)
    theta1 = torch.tensor(np.array([theta1_numpy]))

    result = fHat(theta0, theta1, cand_data_X, cand_data_Y)
    upper_bound = ghat(theta0, theta1, cand_data_X, cand_data_Y, cand_data_T,
                       candidate_ratio, seldonian_type)

    if upper_bound > 0.0:
        result = -10000.0 - upper_bound
    return float(-result)

