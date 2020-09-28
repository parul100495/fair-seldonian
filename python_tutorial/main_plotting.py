from helper import *        # Basic helper functions
import timeit               # To time the execution of ours experiments
from numba import jit, njit       # Just-in-Time (JIT) compiler to accelerate Python code
import math
import pandas as pd
import numpy as np
import sys
import ray
ray.init()
from equation_parser import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy.optimize import minimize
# Folder where the experiment results will be saved
bin_path = 'experiment_results/bin/'


# Generate numPoints data points
@jit(nopython=True)
def generateData(numPoints):
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
@jit(nopython=True)
def predict(theta, x):
	power_value = theta[0] + theta[1] * x
	denominator = 1 + math.exp(-power_value)
	prob_value = 1/denominator
	if prob_value > 0.5:
		return 1
	return 0


# Estimator of the primary objective - negative of the log loss
#@jit(nopython=True)
def fHat(theta, X, Y):
	n = X.size
	predicted_Y = np.empty((n,))
	for i in range(n):
		predicted_Y[i] = predict(theta, X[i])
	res = log_loss(Y, predicted_Y)
	return -res


# Fairness constraint - True positive rate difference less than 0.1
#@jit(nopython=True)
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
#@jit(nopython=True)
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


def simple_logistic(X, Y):
	try:
		X = np.expand_dims(X, axis=1)
		Y = np.expand_dims(Y, axis=1)
		reg = LogisticRegression(solver='lbfgs').fit(X, Y)
		theta0 = reg.intercept_[0]
		theta1 = reg.coef_[0][0]
		return np.array([theta0, theta1])
	except Exception as e:
		print(e)
		return None

# Our Quasi-Seldonian linear regression algorithm operating over data (X,Y).
# The pair of objects returned by QSA is the solution (first element) 
# and a Boolean flag indicating whether a solution was found (second element).
def QSA(X, Y, T, gHats, deltas, ineq):
	candidateData_len = 0.40
	candidateData_X, safetyData_X, candidateData_Y, safetyData_Y = train_test_split(
		X, Y, test_size = 1 - candidateData_len, shuffle = False)
	candidateData_T, safetyData_T = np.split(T, [int(candidateData_len * T.size), ])

	candidateSolution = getCandidateSolution(candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq)
	if candidateSolution:
		passedSafety = safetyTest(candidateSolution, safetyData_X, safetyData_Y, safetyData_T, gHats, deltas, ineq)
		return [candidateSolution, passedSafety]
	else:
		return [None, False]


# Run the safety test on a candidate solution. Returns true if the test is passed.
#   candidateSolution: the solution to test. 
#   (safetyData_X, safetyData_Y): data set D2 to be used in the safety test.
#   (gHats, deltas): vectors containing the behavioral constraints and confidence levels.
def safetyTest(candidateSolution, safetyData_X, safetyData_Y, safetyData_T, gHats, deltas, ineq):
	for i in range(len(gHats)):
		g = gHats[i]
		delta = deltas[i]
		upperBound = g(candidateSolution, safetyData_X, safetyData_Y, safetyData_T, delta, ineq)
		if upperBound > 0.0:
			return False
	return True


# The objective function maximized by getCandidateSolution.
#     thetaToEvaluate: the candidate solution to evaluate.
#     (candidateData_X, candidateData_Y): the data set D1 used to evaluated the solution.
#     (gHats, deltas): vectors containing the behavioral constraints and confidence levels.
#     safetyDataSize: |D2|, used when computing the conservative upper bound on each behavioral constraint.
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
#    (candidateData_X, candidateData_Y): data used to compute a candidate solution.
#    (gHats, deltas): vectors containing the behavioral constraints and confidence levels.
#    safetyDataSize: |D2|, used when computing the conservative upper bound on each behavioral constraint.
def getCandidateSolution(candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq):
	minimizer_method = 'Powell'
	minimizer_options = {'disp': False}
	initialSolution = simple_logistic(candidateData_X, candidateData_Y)
	if initialSolution:
		res = minimize(candidateObjective, x0=initialSolution, method=minimizer_method, options=minimizer_options,
			args=(candidateData_X, candidateData_Y, candidateData_T, gHats, deltas, ineq))
		return res.x
	else:
		return None


@ray.remote
def run_experiments(worker_id, nWorkers, ms, numM, numTrials, mTest, ineq):

	# Results of the Seldonian algorithm runs
	seldonian_solutions_found = np.zeros((numTrials, numM))
	seldonian_failures_g1 = np.zeros((numTrials, numM))
	seldonian_failures_g2 = np.zeros((numTrials, numM))
	seldonian_fs = np.zeros((numTrials, numM))

	# Results of the logistic regression runs
	LS_solutions_found = np.ones((numTrials, numM))
	LS_failures_g1 = np.zeros((numTrials, numM))
	LS_failures_g2 = np.zeros((numTrials, numM))
	LS_fs = np.zeros((numTrials, numM))

	# Prepares file where experiment results will be saved
	experiment_number = worker_id
	outputFile = bin_path + 'results%d.npz' % experiment_number
	print("Writing output to", outputFile)
	
	# Generate the data used to evaluate the primary objective and failure rates
	np.random.seed((experiment_number+1) * 9999)
	(testX, testY, testT) = generateData(mTest)

	for trial in range(numTrials):
		for (mIndex, m) in enumerate(ms):

			# Generate the training data, D
			base_seed = (experiment_number * numTrials)+1
			np.random.seed(base_seed+trial) # done to obtain common random numbers for all values of m			
			(trainX, trainY, trainT)  = generateData(m)

			# Run the logistic regression algorithm
			theta = simple_logistic(trainX, trainY)  # Run least squares linear regression
			if theta:
				trueMSE = -fHat(theta, testX, testY)  # Get the "true" mean squared error using the testData
				LS_failures_g1[
					trial, mIndex] = 1 if trueMSE > 2.0 else 0  # Check if the first behavioral constraint was violated
				LS_failures_g2[
					trial, mIndex] = 1 if trueMSE < 1.25 else 0  # Check if the second behavioral constraint was violated
				LS_fs[ trial, mIndex] = -trueMSE  # Store the "true" negative mean-squared error
				print(
					f"[(worker {worker_id}/{nWorkers}) simple_logistic   trial {trial + 1}/{numTrials}, m {m}] LS fHat over test data: {trueMSE:.10f}" )
			else:
				LS_solutions_found[trial, mIndex] = 0
				LS_failures_g1[
					trial, mIndex] = 1  # Check if the first behavioral constraint was violated
				LS_failures_g2[
					trial, mIndex] = 1 # Check if the second behavioral constraint was violated
				LS_fs[trial, mIndex] = None  # Store the "true" negative mean-squared error
				print(
					f"[(worker {worker_id}/{nWorkers}) Skipping trial because of dirty dataset  trial {trial + 1}/{numTrials}, m {m}]")

				seldonian_solutions_found[trial, mIndex] = 0  # A solution was not found
				seldonian_failures_g1[trial, mIndex] = 0  # Returning NSF means the first constraint was not violated
				seldonian_failures_g2[
					trial, mIndex] = 0  # Returning NSF means the second constraint was not violated
				seldonian_fs[
					trial, mIndex] = None  # This value should not be used later. We use None and later remove the None values
				print(
					f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial + 1}/{numTrials}, m {m}] No solution found")
				continue

			# Run the Quasi-Seldonian algorithm
			(result, passedSafetyTest) = QSA(trainX, trainY, trainT, gHats, deltas, ineq)
			if passedSafetyTest:
				seldonian_solutions_found[trial, mIndex] = 1
				trueMSE = -fHat(result, testX, testY)                               # Get the "true" mean squared error using the testData
				seldonian_failures_g1[trial, mIndex] = 1 if trueMSE > 2.0  else 0   # Check if the first behavioral constraint was violated
				seldonian_failures_g2[trial, mIndex] = 1 if trueMSE < 1.25 else 0	# Check if the second behavioral constraint was violated
				seldonian_fs[trial, mIndex] = -trueMSE                              # Store the "true" negative mean-squared error
				print(f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial+1}/{numTrials}, m {m}] A solution was found: [{result[0]:.10f}, {result[1]:.10f}]\tfHat over test data: {trueMSE:.10f}")
			else:
				seldonian_solutions_found[trial, mIndex] = 0             # A solution was not found
				seldonian_failures_g1[trial, mIndex]     = 0             # Returning NSF means the first constraint was not violated
				seldonian_failures_g2[trial, mIndex]     = 0             # Returning NSF means the second constraint was not violated
				seldonian_fs[trial, mIndex]              = None          # This value should not be used later. We use None and later remove the None values
				print(f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial+1}/{numTrials}, m {m}] No solution found")

				print()

	np.savez(outputFile, 
			 ms=ms, 
			 seldonian_solutions_found=seldonian_solutions_found,
			 seldonian_fs=seldonian_fs, 
			 seldonian_failures_g1=seldonian_failures_g1, 
			 seldonian_failures_g2=seldonian_failures_g2,
			 LS_solutions_found=LS_solutions_found,
			 LS_fs=LS_fs,
			 LS_failures_g1=LS_failures_g1,
			 LS_failures_g2=LS_failures_g2)
	print(f"Saved the file {outputFile}")



if __name__ == "__main__":

	# Create the behavioral constraints: each is a gHat function and a confidence level delta
	gHats = [gHat1, gHat2]
	deltas = [0.05, 0.05]

	if len(sys.argv) < 2:
		print("\nUsage: python main_plotting.py [number_threads]")
		print("       Assuming the default: 8")
		nWorkers = 8  # Workers is the number of threads running experiments in parallel
	else:
		nWorkers = int(sys.argv[1])  # Workers is the number of threads running experiments in parallel
	print(f"Running experiments on {nWorkers} threads")

	# We will use different amounts of data, m. The different values of m will be stored in ms.
	# These values correspond to the horizontal axis locations in all three plots we will make.
	# We will use a logarithmic horizontal axis, so the amounts of data we use shouldn't be evenly spaced.
	ms = [2 ** i for i in
		  range(5, 17)]  # ms = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
	numM = len(ms)

	# How many trials should we average over?
	numTrials = 40  # We pick 70 because with 70 trials per worker, and 16 workers, we get >1000 trials for each value of m

	# How much data should we generate to compute the estimates of the primary objective and behavioral constraint function values
	# that we call "ground truth"? Each candidate solution deemed safe, and identified using limited training data, will be evaluated
	# over this large number of points to check whether it is really safe, and to compute its "true" mean squared error.
	mTest = ms[-1] * 100  # about 5,000,000 test samples

	# Start 'nWorkers' threads in parallel, each one running 'numTrials' trials. Each thread saves its results to a file
	tic = timeit.default_timer()
	_ = ray.get([run_experiments.remote(worker_id, nWorkers, ms, numM, numTrials, mTest, Inequality.T_TEST) for worker_id in
				range(1, nWorkers + 1)])
	toc = timeit.default_timer()
	time_parallel = toc - tic  # Elapsed time in seconds
	print(f"Time elapsed: {time_parallel}")
	ray.shutdown()
