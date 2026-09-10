from qsprpred.data.processing.step import Step
from sklearn.preprocessing import StandardScaler


# Z-score the physchem/protein descriptor columns and leave the fingerprint bits
# as raw 0/1. This reproduces the original Beyond-the-Hype preprocessing.
class ColumnScaler(Step):
    def __init__(self, descriptor_columns):
        self.columns = list(descriptor_columns)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X, y):
        self.scaler.fit(X[self.columns].values)
        self._fitted = True

    def transform(self, X, y):
        assert self._fitted == True, "ColumnScaler has not been fitted yet."
        X[self.columns] = self.scaler.transform(X[self.columns].values)

        # The descriptor frame can carry object-dtype columns, which torch cannot
        # consume. Regular DNNModel does not cast, so do it here rather than patch it.
        return X.astype(float), y
