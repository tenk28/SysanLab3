from additional import *
import numpy as np
from scipy.sparse.linalg import cg

from parameters import Parameters


class Model(Parameters):
    def __init__(self, samples_filename, output_filename, P_dims, polynomials, weights, lambdas, use_custom_function):
        super(Model, self).__init__(samples_filename, output_filename, P_dims, polynomials, weights, lambdas,
                                    use_custom_function)
        self.polynomials = self.get_polynomial_matrix()
        self.lambdas = self.get_lambda_matrix()
        self.psi = self.get_psi_matrix()
        self.a = self.get_a_matrix()
        self.phi = self.get_phi_matrix()
        self.c = self.get_c_matrix()

        self.normalized_results = self.get_normalized_results()
        self.results = self.get_results()
        self.normalized_residuals = self.get_normalized_residuals()
        self.residuals = self.get_residuals()

    def get_polynomial_matrix(self):
        def get_polynomials(X, p):
            polys = list()
            for X_i in X:
                for p_i in np.arange(p + 1):
                    poly = self.get_polynomial(p_i, X_i)
                    if (np.min(poly) != np.max(poly) and
                            (np.min(poly) < 0 or np.max(poly) > 1)):
                        poly = (poly - np.min(poly)) / (np.max(poly) - np.min(poly))
                    polys.append(2 * poly + 1)
            return np.array(polys)

        X1_polynomial = get_polynomials(self.X1_normalized, self.P1_dim)
        X2_polynomial = get_polynomials(self.X2_normalized, self.P2_dim)
        X3_polynomial = get_polynomials(self.X3_normalized, self.P3_dim)
        return np.array((X1_polynomial, X2_polynomial, X3_polynomial))

    def get_lambda_matrix(self):
        def split():
            def sub_split(b):
                if self.use_custom_function is True:
                    lambda1 = self.minimize_system(np.log1p(self.custom_function(self.polynomials[0])), b)
                    lambda2 = self.minimize_system(np.log1p(self.custom_function(self.polynomials[1])), b)
                    lambda3 = self.minimize_system(np.log1p(self.custom_function(self.polynomials[2])), b)
                else:
                    lambda1 = self.minimize_system(np.log1p(self.polynomials[0]), b)
                    lambda2 = self.minimize_system(np.log1p(self.polynomials[1]), b)
                    lambda3 = self.minimize_system(np.log1p(self.polynomials[2]), b)
                return np.hstack((lambda1, lambda2, lambda3))

            lambda_unite = get_lambda(sub_split)
            return lambda_unite

        def unite():
            def sub_unite(b):
                if self.use_custom_function is True:
                    X1_polynomial = np.log1p(self.custom_function(self.polynomials[0].T))
                    X2_polynomial = np.log1p(self.custom_function(self.polynomials[1].T))
                    X3_polynomial = np.log1p(self.custom_function(self.polynomials[2].T))
                else:
                    X1_polynomial = np.log1p(self.polynomials[0].T)
                    X2_polynomial = np.log1p(self.polynomials[1].T)
                    X3_polynomial = np.log1p(self.polynomials[2].T)
                polynomials = np.hstack((X1_polynomial, X2_polynomial, X3_polynomial)).T
                return self.minimize_system(polynomials, b)

            lambda_unite = get_lambda(sub_unite)
            return lambda_unite

        def get_lambda(get_lambda_function):
            lambda_unite = list()
            for b in self.b:
                lambda_unite.append(get_lambda_function(np.log(b + 1)))
            return np.array(lambda_unite)

        if self.lambda_method == LambdaMethod.TRIPLE_SET.name:
            return split()
        else:
            return unite()

    def get_psi_matrix(self):
        def sub_psi(lambda_matrix):
            def Xi_psi(degree, dimensional, polynomial_matrix, _lambda_matrix):
                def psi_columns(_lambda, _polynomial):
                    if self.use_custom_function is True:
                        _psi_column = np.expm1(np.matmul(np.log1p(self.custom_function(_polynomial.T)), _lambda))
                    else:
                        _psi_column = np.expm1(np.matmul(np.log1p(_polynomial.T), _lambda))
                    pc_max = np.max(_psi_column)
                    pc_min = np.min(_psi_column)
                    if (pc_max != pc_min) and (pc_min < 0 or pc_max > 1):
                        _psi_column = (_psi_column - pc_min) / (pc_max - pc_min)
                    elif pc_min < 0:
                        _psi_column = np.zeros_like(_psi_column)
                    elif pc_max > 1:
                        _psi_column = np.ones_like(_psi_column)
                    return _psi_column

                _psi = list()
                _left = 0
                _right = degree + 1
                for _ in np.arange(dimensional):
                    _lambda = _lambda_matrix[_left:_right]
                    polynomial = polynomial_matrix[_left:_right]
                    psi_column = psi_columns(_lambda, polynomial)
                    _psi.append(psi_column)
                    _left = _right
                    _right += degree + 1
                return np.vstack(_psi)

            left = 0
            right = (self.P1_dim + 1) * self.X1_dim
            x1_psi = Xi_psi(self.P1_dim, self.X1_dim, self.polynomials[0], lambda_matrix[left:right])
            left = right
            right = left + (self.P2_dim + 1) * self.X2_dim
            x2_psi = Xi_psi(self.P2_dim, self.X2_dim, self.polynomials[1], lambda_matrix[left:right])
            left = right
            right = left + (self.P3_dim + 1) * self.X3_dim
            x3_psi = Xi_psi(self.P3_dim, self.X3_dim, self.polynomials[2], lambda_matrix[left:right])
            return np.array((x1_psi, x2_psi, x3_psi), dtype=object)

        psi = list()
        for lambda_k in self.lambdas:
            psi.append(sub_psi(lambda_k))
        return np.array(psi)

    def get_a_matrix(self):
        def sub_a(psi, Y):
            a_i = list()
            for sub_psi in psi:
                if self.use_custom_function is True:
                    matrix_a = np.log1p(self.custom_function(sub_psi.astype(float)))
                else:
                    matrix_a = np.log1p(sub_psi.astype(float))
                matrix_b = np.log1p(Y)
                a_i.append(self.minimize_system(matrix_a, matrix_b))
            return np.hstack(a_i)

        a = list()
        for i in np.arange(self.Y_dim):
            a.append(sub_a(self.psi[i], self.Y_normalized[i]))
        return np.array(a)

    def get_phi_matrix(self):
        def sub_phi(psi, a):
            def phi_columns(psi_i, a_i):
                psi_i = psi_i.astype(float)
                if self.use_custom_function is True:
                    phi_column = np.expm1(np.matmul(np.log1p(self.custom_function(psi_i.T)), a_i))
                else:
                    phi_column = np.expm1(np.matmul(np.log1p(psi_i.T), a_i))
                pc_min = np.min(phi_column)
                pc_max = np.max(phi_column)
                if (pc_min != pc_max) and (pc_min < 0 or pc_max > 0):
                    phi_column = (phi_column - pc_min) / (pc_max - pc_min)
                elif pc_min < 0:
                    phi_column = np.zeros_like(phi_column)
                elif pc_max > 1:
                    phi_column = np.ones_like(phi_column)
                return phi_column

            left = 0
            right = self.X1_dim
            X1_phi = phi_columns(psi[0], a[left:right])

            left = right
            right += self.X2_dim
            X2_phi = phi_columns(psi[1], a[left:right])

            left = right
            right += self.X3_dim
            X3_phi = phi_columns(psi[2], a[left:right])

            return np.array((X1_phi, X2_phi, X3_phi))

        phi = list()
        for i in np.arange(self.Y_dim):
            phi.append(sub_phi(self.psi[i], self.Y_normalized[i]))
        return np.array(phi)

    def get_c_matrix(self):
        def sub_c(phi_i, y_i):
            if self.use_custom_function is True:
                c_i = self.minimize_system(np.log1p(self.custom_function(phi_i)), np.log1p(y_i))
            else:
                c_i = self.minimize_system(np.log1p(phi_i), np.log1p(y_i))
            return c_i

        c = list()
        for i in np.arange(self.Y_dim):
            c.append(sub_c(self.phi[i], self.Y_normalized[i]))
        return np.array(c)

    def get_normalized_results(self):
        normalized_results = list()
        for i in range(self.Y_dim):
            n_r = np.dot(self.phi[i].T, self.c[i])
            nr_min = np.min(n_r)
            nr_max = np.max(n_r)
            if nr_min < 0 or nr_max > 1:
                n_r = (n_r - nr_min) / (nr_max - nr_min)
            n_r = self.HONESTY * n_r + (1 - self.HONESTY) * self.Y_normalized[i]
            normalized_results.append(n_r)
        return np.array(normalized_results)

    def get_results(self):
        results = np.copy(self.normalized_results)
        for i in np.arange(self.Y_dim):
            Y_max = np.max(self.Y[i])
            Y_min = np.min(self.Y[i])
            results[i] = results[i] * (Y_max - Y_min) + Y_min
        return results

    def get_normalized_residuals(self):
        normalized_residuals = list()
        for i in np.arange(self.Y_dim):
            n_res = np.max(np.abs(self.Y_normalized[i] - self.normalized_results[i]))
            normalized_residuals.append(n_res)
        return np.array(normalized_residuals)

    def get_residuals(self):
        residuals = list()
        for i in np.arange(self.Y_dim):
            res = np.max(np.abs(self.Y[i] - self.results[i]))
            residuals.append(res)
        return np.array(residuals)

    def minimize_system(self, A, b):
        A = A.T
        b = np.matmul(A.T, b)
        A = np.matmul(A.T, A)
        X, _ = cg(A, b, tol=self.EPS)
        return X

    @staticmethod
    def custom_function(x):
        return 2 * np.sin(x)
