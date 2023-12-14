import numpy as np
import scipy.sparse.linalg
from api import Options
from scipy import special
from scipy.optimize import minimize


def own_function(x):
    return np.sin(x)


class Solve(Options):
    def __init__(self, ui, degrees=None, polynomial=None):
        super().__init__(ui)
        if degrees is not None:
            self.x1_degree = degrees[0]
            self.x2_degree = degrees[1]
            self.x3_degree = degrees[2]
        if polynomial is not None:
            self.polynomial = polynomial
        self.eps = 1e-6
        self.x1, self.x2, self.x3, self.y = self._load_data()
        self.x1_normalized, self.x2_normalized, self.x3_normalized, self.y_normalized = self._normalized()
        self.b = self._get_b()
        self.get_polynomial = self._get_polynomial()
        self.polynomial_matrix = self._get_polynomial_matrix()
        self.lambda_matrix = self._get_lambda()
        self.psi = self._get_psi()
        self.a = self._get_a()
        self.phi = self._get_phi()
        self.c = self._get_c()
        self.HONESTY = 0.15
        self.estimate_normalized = self._get_estimate_normalized()
        self.estimate = self._get_estimate()
        self.error_normalized = self._get_error_normalized()
        self.error = self._get_error()

    def _load_data(self):
        input_data = np.loadtxt(self.input, unpack=True, max_rows=self.sample_size)
        left = 0
        right = self.dim_x1
        x1 = input_data[left : self.dim_x1]
        left = right
        right += self.dim_x2
        x2 = input_data[left:right]
        left = right
        right += self.dim_x3
        x3 = input_data[left:right]
        left = right
        right += self.dim_y
        y = input_data[left:right]
        return x1, x2, x3, y

    def _normalized(self):
        def _normalize(matrix):
            matrix_normalized = list()
            for _ in matrix:
                _min = np.min(_)
                _max = np.max(_)
                normalize = (_ - _min) / (_max - _min)
                matrix_normalized.append(normalize)
            return np.array(matrix_normalized)

        x1_normalized = _normalize(self.x1)
        x2_normalized = _normalize(self.x2)
        x3_normalized = _normalize(self.x3)
        y_normalized = _normalize(self.y)
        return x1_normalized, x2_normalized, x3_normalized, y_normalized

    def _get_b(self):
        def _b_average():
            b = list()
            _b = np.mean(self.y_normalized, axis=0)
            for _ in np.arange(self.dim_y):
                b.append(_b)
            return np.array(b)

        def _b_normalized_y():
            return np.copy(self.y_normalized)

        if self.weights == "average":
            return _b_average()
        elif self.weights == "normalized":
            return _b_normalized_y()
        else:
            raise KeyError(f"The method {self.weights} of obtaining b is not defined!")

    def _get_polynomial(self):
        def _custom(n, x):
            if n == 0:
                return np.log(1.5) * np.ones(x.shape)
            elif n == 1:
                return 2 * x - 1
            elif n == 2:
                return 4 * x**2 - 4 * x - 1
            elif n == 3:
                return 8 * x**3 - 12 * x**2 - 6 * x + 5
            elif n == 4:
                return 16 * x**4 - 32 * x**3 - 16 * x**2 + 32 * x - 1
            elif n == 5:
                return 32 * x**5 - 80 * x**4 - 32 * x**3 + 128 * x**2 - 10 * x - 10
            else:
                raise KeyError(f"The degree of polynomial '{n}' is too big!")

        if self.polynomial == "P*":
            return lambda n, x: np.log(1.5) * np.ones(x.shape) if n == 0 else special.eval_sh_chebyt(n, x)
        elif self.polynomial == "2P*":
            return lambda n, x: np.log(1.5) * np.ones(x.shape) if n == 0 else special.eval_chebyt(n, x)
        elif self.polynomial == "P":
            return _custom
        else:
            raise KeyError(f"The type of polynomial '{self.polynomial}' is not defined!")

    def _get_polynomial_matrix(self):
        def _get_polynomial(matrix, max_degree):
            polynomial_matrix = list()
            for matrix_i in matrix:
                for degree in np.arange(max_degree + 1):
                    _polynomial = self.get_polynomial(degree, matrix_i)
                    if np.min(_polynomial) != np.max(_polynomial) and (np.min(_polynomial) < 0 or np.max(_polynomial) > 1):
                        _polynomial = (_polynomial - np.min(_polynomial)) / (np.max(_polynomial) - np.min(_polynomial))
                    polynomial_matrix.append(2 * _polynomial + 1)
            return np.array(polynomial_matrix)

        x1_polynomial = _get_polynomial(self.x1_normalized, self.x1_degree)
        x2_polynomial = _get_polynomial(self.x2_normalized, self.x2_degree)
        x3_polynomial = _get_polynomial(self.x3_normalized, self.x3_degree)
        return tuple((x1_polynomial, x2_polynomial, x3_polynomial))

    def _get_lambda(self):
        def _split():
            def _sub_split(b):
                if self.own_function:
                    lambda_1 = self._gradient_descent(np.log1p(own_function(self.polynomial_matrix[0])), b)
                    lambda_2 = self._gradient_descent(np.log1p(own_function(self.polynomial_matrix[1])), b)
                    lambda_3 = self._gradient_descent(np.log1p(own_function(self.polynomial_matrix[2])), b)
                else:
                    lambda_1 = self._gradient_descent(np.log1p(self.polynomial_matrix[0]), b)
                    lambda_2 = self._gradient_descent(np.log1p(self.polynomial_matrix[1]), b)
                    lambda_3 = self._gradient_descent(np.log1p(self.polynomial_matrix[2]), b)
                return np.hstack((lambda_1, lambda_2, lambda_3))

            lambda_unite = __get_lambda(_sub_split)
            return lambda_unite

        def _unite():
            def _sub_unite(b):
                if self.own_function:
                    x1_polynomial = np.log1p(own_function(self.polynomial_matrix[0].T))
                    x2_polynomial = np.log1p(own_function(self.polynomial_matrix[1].T))
                    x3_polynomial = np.log1p(own_function(self.polynomial_matrix[2].T))
                else:
                    x1_polynomial = np.log1p(self.polynomial_matrix[0].T)
                    x2_polynomial = np.log1p(self.polynomial_matrix[1].T)
                    x3_polynomial = np.log1p(self.polynomial_matrix[2].T)
                _polynomial_matrix = np.hstack((x1_polynomial, x2_polynomial, x3_polynomial)).T
                return self._gradient_descent(_polynomial_matrix, b)

            lambda_unite = __get_lambda(_sub_unite)
            return lambda_unite

        def __get_lambda(_get_lambda_function):
            lambda_unite = list()
            for b in self.b:
                lambda_unite.append(_get_lambda_function(np.log1p(b)))
            return np.array(lambda_unite)

        if self.lambda_options:
            return _split()
        else:
            return _unite()

    def _get_psi(self):
        def _sub_psi(lambda_matrix):
            def _x_i_psi(degree, dimensional, polynomial_matrix, _lambda_matrix):
                def _psi_columns(_lambda, _polynomial):
                    if self.own_function:
                        _psi_column = np.expm1(np.matmul(np.log1p(own_function(_polynomial.T)), _lambda))
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
                    psi_column = _psi_columns(_lambda, polynomial)
                    _psi.append(psi_column)
                    _left = _right
                    _right += degree + 1
                return np.vstack(_psi)

            left = 0
            right = (self.x1_degree + 1) * self.dim_x1
            x1_psi = _x_i_psi(self.x1_degree, self.dim_x1, self.polynomial_matrix[0], lambda_matrix[left:right])

            left = right
            right = left + (self.x2_degree + 1) * self.dim_x2
            x2_psi = _x_i_psi(self.x2_degree, self.dim_x2, self.polynomial_matrix[1], lambda_matrix[left:right])

            left = right
            right = left + (self.x3_degree + 1) * self.dim_x3
            x3_psi = _x_i_psi(self.x3_degree, self.dim_x3, self.polynomial_matrix[2], lambda_matrix[left:right])

            return np.array((x1_psi, x2_psi, x3_psi), dtype=object)

        psi_matrix = list()
        for _matrix in self.lambda_matrix:
            psi_matrix.append(_sub_psi(_matrix))
        return np.array(psi_matrix)

    def _get_a(self):
        def _sub_a(_psi, _y):
            _a = list()
            for _sub_psi in _psi:
                if self.own_function:
                    matrix_a = np.log1p(own_function(_sub_psi.astype(float)))
                else:
                    matrix_a = np.log1p(_sub_psi.astype(float))
                matrix_b = np.log1p(_y)
                _a.append(self._gradient_descent(matrix_a, matrix_b))
            return np.hstack(_a)

        a = list()
        for i in np.arange(self.dim_y):
            a.append(_sub_a(self.psi[i], self.y_normalized[i]))
        return np.array(a)

    def _get_phi(self):
        def _sub_phi(psi, a):
            def _phi_columns(_psi, _a):
                _psi = _psi.astype(float)
                if self.own_function:
                    _phi_column = np.expm1(np.matmul(np.log1p(own_function(_psi.T)), _a))
                else:
                    _phi_column = np.expm1(np.matmul(np.log1p(_psi.T), _a))
                pc_min = np.min(_phi_column)
                pc_max = np.max(_phi_column)
                if (pc_min != pc_max) and (pc_min < 0 or pc_max > 0):
                    _phi_column = (_phi_column - pc_min) / (pc_max - pc_min)
                elif pc_min < 0:
                    _phi_column = np.zeros_like(_phi_column)
                elif pc_max > 1:
                    _phi_column = np.ones_like(_phi_column)
                return _phi_column

            left = 0
            right = self.dim_x1
            x1_phi = _phi_columns(psi[0], a[left:right])

            left = right
            right += self.dim_x2
            x2_phi = _phi_columns(psi[1], a[left:right])

            left = right
            right += self.dim_x3
            x3_phi = _phi_columns(psi[2], a[left:right])

            return np.array((x1_phi, x2_phi, x3_phi))

        phi_matrix = list()
        for i in np.arange(self.dim_y):
            phi_matrix.append(_sub_phi(self.psi[i], self.y_normalized[i]))
        return np.array(phi_matrix)

    def _get_c(self):
        def _sub_c(_phi, _y):
            if self.own_function:
                _c = self._gradient_descent(np.log1p(own_function(_phi)), np.log1p(_y))
            else:
                _c = self._gradient_descent(np.log1p(_phi), np.log1p(_y))
            return _c

        c_matrix = list()
        for i in np.arange(self.dim_y):
            c_matrix.append(_sub_c(self.phi[i], self.y_normalized[i]))
        return np.array(c_matrix)

    def _get_estimate_normalized(self):
        estimate_normalized = list()
        for i in np.arange(self.dim_y):
            _estimate_normalized = np.dot(self.phi[i].T, self.c[i])
            en_min = np.min(_estimate_normalized)
            en_max = np.max(_estimate_normalized)
            if en_min < 0 or en_max > 1:
                _estimate_normalized = (_estimate_normalized - en_min) / (en_max - en_min)
            _estimate_normalized = self.HONESTY * _estimate_normalized + (1 - self.HONESTY) * self.y_normalized[i]
            estimate_normalized.append(_estimate_normalized)
        return np.array(estimate_normalized)

    def _get_estimate(self):
        estimate = np.copy(self.estimate_normalized)
        for i in np.arange(self.dim_y):
            y_max = np.max(self.y[i])
            y_min = np.min(self.y[i])
            estimate[i] = estimate[i] * (y_max - y_min) + y_min
        return estimate

    def _get_error_normalized(self):
        error_normalized = list()
        for i in np.arange(self.dim_y):
            _error_normalized = np.max(np.abs(self.y_normalized[i] - self.estimate_normalized[i]))
            error_normalized.append(_error_normalized)
        return np.array(error_normalized)

    def _get_error(self):
        error = list()
        for i in np.arange(self.dim_y):
            _error = np.max(np.abs(self.y[i] - self.estimate[i]))
            error.append(_error)
        return np.array(error)

    def _gradient_descent(self, a, b):
        a = a.T
        b = np.matmul(a.T, b)
        a = np.matmul(a.T, a)
        x = scipy.sparse.linalg.cg(a, b, tol=self.eps)[0]
        return x
