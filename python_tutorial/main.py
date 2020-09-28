from helper import *
from equation_parser import *
import math
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import t
from sklearn.metrics import log_loss
from scipy.optimize import minimize # The black-box optimization algorithm used to find a candidate solution

# Generate numPoints data points
def generateSyntheticData(numPoints):
	X = np.random.normal(0.0, 1.0, numPoints)
	Y_setting = X + np.random.normal(-1.0, 1.0, numPoints)
	Y = np.empty((numPoints,))
	for i in range(numPoints):
		if Y_setting[i] >= 0.5:
			Y[i] = 1
		else:
			Y[i] = 0
	T = np.empty((numPoints,))
	T[::2] = 1
	T[1::2] = 0
	return (X, Y, T)


# simple logistic regression classifier
def predict(theta, x):
	power_value = theta[0] + theta[1] * x
	denominator = 1 + math.exp(-power_value)
	prob_value = 1/denominator
	if prob_value > 0.5:
		return 1
	return 0


# Estimator of the primary objective - negative of the log loss
def fHat(theta, X, Y):
	n = X.size
	predicted_Y = np.empty((n,))
	for i in range(n):
		predicted_Y[i] = predict(theta, X[i])
	res = log_loss(Y, predicted_Y)
	return -res


# Fairness constraint - True positive rate difference less than 0.1
def gHat1(theta, X, Y, T, delta, ineq):
	n = X.size
	predicted_Y = np.empty((n,))
	for i in range(n):
		predicted_Y[i] = predict(theta, X[i])
	rev_polish_notation = "TP(0.0) TP(1.0) - abs 0.1 -"
	r = construct_expr_tree(rev_polish_notation)
	_, u = eval_expr_tree_conf_interval(r, pd.Series(Y), pd.Series(predicted_Y), pd.Series(T), delta, ineq)
	# res = u - 0.1
	return u

# Fairness constraint - True positive rate difference greater than -0.1
def gHat2(theta, X, Y, T, delta, ineq):
	n = X.size
	predicted_Y = np.empty((n,))
	for i in range(n):
		predicted_Y[i] = predict(theta, X[i])
	rev_polish_notation = "-0.1 TP(0.0) TP(1.0) - abs -"
	r = construct_expr_tree(rev_polish_notation)
	_, u = eval_expr_tree_conf_interval(r, pd.Series(Y), pd.Series(predicted_Y), pd.Series(T), delta, ineq)
	# res = -0.1 - u
	return u


# Our Quasi-Seldonian linear regression classifier
def QSA(X, Y, T, gHats, deltas, ineq):
	candidateData_len = 0.40
	candidateData_X, safetyData_X, candidateData_Y, safetyData_Y = train_test_split(
								X, Y, test_size=1-candidateData_len, shuffle=False)
	candidateData_T, safetyData_T = np.split(T, [int(candidateData_len*T.size), ])

	candidateSolution = getCandidateSolution(candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq)
	passedSafety = safetyTest(candidateSolution, safetyData_X, safetyData_Y, safetyData_T, gHats, deltas, ineq)
	return [candidateSolution, passedSafety]


# Run the safety test on a candidate solution. Returns true if the test is passed.
def safetyTest(candidateSolution, safetyData_X, safetyData_Y, safetyData_T, gHats, deltas, ineq):
	for i in range(len(gHats)):
		g = gHats[i]
		delta = deltas[i]
		upperBound = g(candidateSolution, safetyData_X, safetyData_Y, safetyData_T, delta, ineq)
		if upperBound > 0.0:
			return False
	return True


# The objective function maximized by getCandidateSolution.
def candidateObjective(thetaToEvaluate, candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq):
	result = fHat(thetaToEvaluate, candidateData_X, candidateData_Y)
	predictSafetyTest = True
	for i in range(len(gHats)):
		g = gHats[i]
		delta = deltas[i]
		upperBound = g(thetaToEvaluate, candidateData_X, candidateData_Y, candidateData_T, delta, ineq)
		if upperBound > 0.0:
			if predictSafetyTest:
				predictSafetyTest = False
				result = -100000.0
			result = result - upperBound
	return -result


# Use the provided data to get a candidate solution expected to pass the safety test.
def getCandidateSolution(candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq):
	minimizer_method = 'Powell'
	minimizer_options = {'disp': False}
	initialSolution = simple_logistic(candidateData_X, candidateData_Y)
	res = minimize(candidateObjective, x0=initialSolution, method=minimizer_method, options=minimizer_options, 
		args=(candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq))
	return res.x


def simple_logistic(X, Y):
	X = np.expand_dims(X, axis=1)
	Y = np.expand_dims(Y, axis=1)
	reg = LogisticRegression().fit(X, Y)
	theta0 = reg.intercept_[0]
	theta1 = reg.coef_[0][0]
	return np.array([theta0, theta1])


if __name__ == "__main__":
	np.random.seed(0)
	numPoints = 5000
	(X, Y, T) = generateSyntheticData(numPoints)
	# Create the fairness constraints and a confidence level delta
	gHats = [gHat1, gHat2]  # The 1st gHat requires TPR < 0.1. The 2nd gHat requires TPR > -0.1
	deltas = [0.05, 0.05]
	print("With T_TEST:")
	(result, found) = QSA(X, Y, T, gHats, deltas, Inequality.T_TEST)
	if found:
		print("A solution was found: [%.10f, %.10f]" % (result[0], result[1]))
		print("fHat of solution (computed over all data, D): ", fHat(result, X, Y))
	else:
		print("No solution found")
	print("With HOEFFDING_INEQUALITY:")
	(result1, found) = QSA(X, Y, T, gHats, deltas, Inequality.HOEFFDING_INEQUALITY)
	if found:
		print("A solution was found: [%.10f, %.10f]" % (result1[0], result1[1]))
		print("fHat of solution (computed over all data, D): ", fHat(result1, X, Y))
	else:
		print("No solution found")
