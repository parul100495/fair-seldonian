from sklearn.model_selection import train_test_split
from logistic_regression_functions import *
from scipy.optimize import minimize
import numpy as np
import torch
candidate_ratio = 0.40


def QSA(X, Y, T, seldonian_type, init_sol, init_sol1):
    """
    This function is used to run the qsa (Quasi-Seldonian Algorithm)

    :param X: The features of the dataset
    :param Y: The corresponding labels of the dataset
    :param T: The corresponding sensitive attributes of the dataset
    :param seldonian_type: The mode used in the experiment
    :param init_sol: The initial theta values for the model
    :param init_sol1: The additional initial theta values for the model
    :return: (theta, theta1, passed_safety) tuple containing optimal theta values and bool whether the candidate
    solution passed safety test or not.
    """
    cand_data_X, safe_data_X, cand_data_Y, safe_data_Y = train_test_split(X, Y,
                                                                          test_size = 1 - candidate_ratio,
                                                                          shuffle = False)
    cand_data_T, safe_data_T = np.split(T, [int(candidate_ratio * T.size),])

    theta, theta1 = get_cand_solution(cand_data_X, cand_data_Y, cand_data_T, candidate_ratio, seldonian_type, init_sol, init_sol1)
    print("Actual cand sol upperbound: ", eval_ghat(theta, theta1,
                                                    cand_data_X, cand_data_Y, cand_data_T,
                                                    seldonian_type))
    passed_safety = safety_test(theta, theta1, safe_data_X, safe_data_Y, safe_data_T, seldonian_type)
    return [theta, theta1, passed_safety]


def safety_test(theta, theta1, safe_data_X, safe_data_Y, safe_data_T, seldonian_type):
    """
    This function does the safety test.

    :param theta: The optimal theta values for the model
    :param theta1: The additional optimal theta values for the model
    :param safe_data_X: The features of the safety dataset
    :param safe_data_Y: The corresponding labels of the safety dataset
    :param safe_data_T: The corresponding sensitive attributes of the safety dataset
    :param seldonian_type: The mode used in the experiment
    :return: Bool value of whether the candidate solution passed safety test or not.
    """
    upper_bound = eval_ghat(theta, theta1, safe_data_X, safe_data_Y, safe_data_T, seldonian_type)
    print("Safety test upperbound: ", upper_bound)
    if upper_bound > 0.0:
        return False
    return True


def get_cand_solution(cand_data_X, cand_data_Y, cand_data_T, candidate_ratio,
                      seldonian_type, init_sol, init_sol1):
    """
    This function provides the candidate solution.

    :param cand_data_X: The features of the candidate dataset
    :param cand_data_Y: The corresponding labels of the candidate dataset
    :param cand_data_T: The corresponding sensitive attributes of the candidate dataset
    :param seldonian_type: The mode used in the experiment
    :param init_sol: The initial theta values for the model
    :param init_sol1: The additional initial theta values for the model
    :return: The candidate solution (theta, theta1).
    """
    if init_sol is None:
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
    """
    This function calculates the value of the objective function which would be
    minimized by the optimizer.

    :param theta: The theta values for the model
    :param cand_data_X: The features of the candidate dataset
    :param cand_data_Y: The corresponding labels of the candidate dataset
    :param cand_data_T: The corresponding sensitive attributes of the candidate dataset
    :param candidate_ratio: The candidate:safety ratio used in the experiment
    :param seldonian_type: The mode used in the experiment
    :return: The objective value.
    """
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


def _get_cand_solution2(cand_data_X, cand_data_Y, cand_data_T, candidate_ratio, seldonian_type):
    init_sol, init_sol1 = simple_logistic(cand_data_X, cand_data_Y)
    init_fhat = fHat(init_sol, init_sol1, cand_data_X, cand_data_Y)
    init_ghat = eval_ghat(init_sol, init_sol1, cand_data_X, cand_data_Y, cand_data_T,
                          seldonian_type)
    init_fhat.backward()
    numerator = init_sol.grad + init_sol1.grad
    init_ghat.backward()
    denominator = init_sol.grad + init_sol1.grad
    lambda_value = -numerator/denominator
    fin_lambda = None
    for i in range(len(init_sol + 1)):
        if lambda_value[i] > 0:
            fin_lambda = float(lambda_value[i])
            break
    if not fin_lambda:
        fin_lambda = 1
    print("Initial LS upperbound: ", eval_ghat(init_sol, init_sol1,
                                               cand_data_X, cand_data_Y, cand_data_T,
                                               seldonian_type))
    theta = init_sol.detach().numpy()
    theta1 = init_sol1.detach().numpy()
    init_theta = np.concatenate((theta, theta1))
    res = minimize(cand_obj2, x0 = init_theta, method = 'BFGS',
                     options = {'disp': False, 'maxiter': 12000},
                     args = (cand_data_X, cand_data_Y, cand_data_T, candidate_ratio, seldonian_type, fin_lambda))
    # ndarray -> tensor of theta
    theta_numpy = res.x[:-1]
    theta1_numpy = res.x[-1]
    theta0 = torch.tensor(theta_numpy)
    theta1 = torch.tensor(np.array([theta1_numpy]))
    return theta0, theta1


def _cand_obj2(theta, cand_data_X, cand_data_Y, cand_data_T, candidate_ratio, seldonian_type, lambda_value):
    theta_numpy = theta[:-1]
    theta1_numpy = theta[-1]
    theta0 = torch.tensor(theta_numpy)
    theta1 = torch.tensor(np.array([theta1_numpy]))

    result = fHat(theta0, theta1, cand_data_X, cand_data_Y)
    upper_bound = eval_ghat(theta0, theta1, cand_data_X, cand_data_Y, cand_data_T,
                            seldonian_type)
    if upper_bound > 0:
        result = float(-1000 - (lambda_value * upper_bound))
    else:
        result = float(-result - (lambda_value * upper_bound))
    return float(-result)
