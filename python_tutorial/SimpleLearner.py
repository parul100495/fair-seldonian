import numpy as np
import pandas as pd
# Author: Parul Gupta

class LeastSquaresBinaryClassifierLearner:
    def __init__(self):
        self.weights = None


    def fit(self, X, Y):
        self.lsqinfo = np.linalg.lstsq(X, Y, rcond=-1)
        self.weights = pd.Series(self.lsqinfo[0], index=list(X))


    def predict(self, X):
        pred = X.dot(np.asarray(self.weights))
        return 1 * (pred > 0.5)
