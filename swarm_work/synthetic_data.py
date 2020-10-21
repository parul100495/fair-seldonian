from random import random, seed
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


def get_data(N, features, t_ratio, tp0_ratio, tp1_ratio, random_seed):
    random_state = int(random_seed * 99) + 1
    seed(random_state)
    T = np.random.default_rng(random_state).binomial(1, t_ratio, N)
    A = np.zeros(T.shape)
    Y = np.zeros(T.shape)
    X = np.zeros(T.shape)
    group0_X = A[T.astype(str) == '0']
    T0_Y = np.random.default_rng(random_state).binomial(1, tp0_ratio, group0_X.shape)
    group1_X = A[T.astype(str) == '1']
    T1_Y = np.random.default_rng(random_state).binomial(1, tp1_ratio, group1_X.shape)
    j = 0  # for 0
    k = 0  # for 1
    for i in range(T.shape[0]):
        if T[i] == 0:
            # get from T0_Y
            Y[i] = T0_Y[j]
            X[i] = Y[i] * random()

            j += 1
        elif T[i] == 1:
            # get from T1_Y
            Y[i] = T1_Y[k]
            X[i] = Y[i] * random()
            k += 1

    T = pd.Series(T)
    X1 = np.random.rand(N, features-2)
    # X = pd.DataFrame(X)
    X = pd.concat([pd.DataFrame(X), pd.DataFrame(X1), T], axis = 1)
    Y = pd.Series(Y)
    # All = pd.concat([X, Y, T], axis = 1)
    return X, Y, T


# Generate numPoints data points
# def generateData(frac, All, random_state, mTest):
#     all_train, all_test, y_train, y_test = train_test_split(All, All.iloc[:, -1], test_size=mTest, random_state=42)
#     # test dataset
#     test_tmp = all_test.iloc[:, :-1]
#     T_test = all_test.iloc[:, -1]
#     X_test = test_tmp.iloc[:, :-1]
#     Y_test = test_tmp.iloc[:, -1]
#
#     # train dataset
#     subsampling = all_train.sample(frac=frac, random_state=random_state)
#     subsampling = subsampling.reset_index()
#     subsampling = subsampling.drop(columns=['index'])
#     tmp = subsampling.iloc[:, :-1]
#     T = subsampling.iloc[:, -1]
#     X = tmp.iloc[:, :-1]
#     Y = tmp.iloc[:, -1]
#     return np.array(X_test), np.array(Y_test), np.array(T_test), np.array(X), np.array(Y), np.array(T)

