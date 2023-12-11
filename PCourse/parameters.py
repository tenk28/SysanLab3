import pandas as pd
import numpy as np
from scipy import special

from additional import *


class Parameters:
    def __init__(self, samples_filename, output_filename, P_dims, polynomials, weights, lambdas, use_custom_function):
        self.get_polynomial = self.get_polynomial_method(polynomials)
        self.weight_method = weights
        self.lambda_method = lambdas
        self.use_custom_function = use_custom_function

        self.samples_filename = samples_filename
        self.output_filename = output_filename

        self.samples = pd.read_excel(self.samples_filename, index_col=0)

        self.Q, self.X_dims, self.X1_dim, self.X2_dim, self.X3_dim, self.Y_dim = self.get_data_dims()
        self.P1_dim, self.P2_dim, self.P3_dim = [P_i for P_i in P_dims]
        self.P_dims = np.array((self.P1_dim, self.P2_dim, self.P3_dim))

        self.X1, self.X2, self.X3, self.Y = self.get_data()
        self.X1_normalized, self.X2_normalized, self.X3_normalized, self.Y_normalized = self.get_normalized_data()
        self.b = self.get_b()

        self.EPS = 1e-12
        self.HONESTY = 0.15

    def get_data_dims(self):
        Q = len(self.samples.index)
        X1_dim, X2_dim, X3_dim, Y_dim = 0, 0, 0, 0
        for column in self.samples.columns:
            variable_name = column[0]
            index = column[1]
            if variable_name == 'x':
                if index == '1':
                    X1_dim += 1
                elif index == '2':
                    X2_dim += 1
                elif index == '3':
                    X3_dim += 1
            elif variable_name == 'y':
                Y_dim += 1
        X_dims = np.array((X1_dim, X2_dim, X3_dim))
        return Q, X_dims, X1_dim, X2_dim, X3_dim, Y_dim

    def get_data(self):
        X = self.samples.iloc[:, :self.X1_dim + self.X2_dim + self.X3_dim].to_numpy()

        X1 = X[:, :self.X1_dim]
        X2 = X[:, self.X1_dim:self.X1_dim + self.X2_dim]
        X3 = X[:, self.X1_dim + self.X2_dim:self.X1_dim + self.X2_dim + self.X3_dim]
        Y = self.samples.iloc[:, self.X1_dim + self.X2_dim + self.X3_dim:].to_numpy()
        return X1.T, X2.T, X3.T, Y.T

    @staticmethod
    def normalize(matrix):
        matrix_normalized = list()
        for _ in matrix:
            _min = np.min(_)
            _max = np.max(_)
            normalize = (_ - _min) / (_max - _min)
            matrix_normalized.append(normalize)
        return np.array(matrix_normalized)

    def get_normalized_data(self):
        X1_normalized = self.normalize(self.X1)
        X2_normalized = self.normalize(self.X2)
        X3_normalized = self.normalize(self.X3)
        Y_normalized = self.normalize(self.Y)
        return X1_normalized, X2_normalized, X3_normalized, Y_normalized

    def get_b(self):
        def normalized():
            return np.copy(self.Y_normalized)

        def max_min():
            b = list()
            means = np.max(self.Y_normalized, axis=0) - np.min(self.Y_normalized, axis=0)
            for _ in np.arange(self.Y_dim):
                b.append(means)
            return np.array(b)

        if self.weight_method == WeightMethod.NORMED.name:
            return normalized()
        elif self.weight_method == WeightMethod.MIN_MAX.name:
            return max_min()

    @staticmethod
    def get_polynomial_method(polynomials):
        if polynomials == PolynomialMethod.SHIFTED_LEGANDRE.name:
            return lambda n, x: np.log(1.5) * np.ones(x.shape) if n == 0 else special.eval_sh_legendre(n, x)
        elif polynomials == PolynomialMethod.DOUBLED_SHIFTED_LEGANDRE.name:
            return lambda n, x: np.log(1.5) * np.ones(x.shape) if n == 0 else 2 * special.eval_sh_legendre(n, x)
        else:  # LEGANDRE
            return lambda n, x: np.log(1.5) * np.ones(x.shape) if n == 0 else special.eval_legendre(n, x)

    @staticmethod
    def get_small_phi_by_variant(n, x):
        return special.eval_sh_legendre(n, x)

    @staticmethod
    def get_small_custom_phi(n, x):
        phi = np.arctan(np.power(x, n))
        return phi
