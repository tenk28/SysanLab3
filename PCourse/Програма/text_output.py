import numpy as np


class Output:
    @staticmethod
    def _show_lambda(solver, y_i):
        out = ""
        first = (solver.x1_degree+1) * solver.dim_x1
        lambda_1 = solver.lambda_matrix[y_i, :first]
        out += f"Лямбда 1:\n{lambda_1}\n\n"
        second = (solver.x2_degree+1) * solver.dim_x2 + first
        lambda_2 = solver.lambda_matrix[y_i, first:second]
        out += f"Лямбда 2:\n{lambda_2}\n\n"
        third = (solver.x3_degree+1) * solver.dim_x3 + second
        lambda_3 = solver.lambda_matrix[y_i, second:third]
        out += f"Лямбда 3:\n{lambda_3}\n\n"
        return out

    @staticmethod
    def _show_psi(psi, dim_y):
        out = ""
        for i in (dim_y,):
            sub_psi = np.vstack(psi[i]).T
            out += f"Псі:\n{sub_psi}\n\n"
        return out

    @staticmethod
    def _show_a(a, y_i):
        out = f"A:\n{a[y_i].T}\n\n"
        return out

    @staticmethod
    def _show_c(c, y_i):
        out = f"C:\n{c[y_i].T}\n\n"
        return out

    @staticmethod
    def __get_coefficient(a, c, lambda_matrix, dim, degrees, i, j, p):
        _dim = dim[:i - 1]
        _deg = degrees[:i - 1] + 1
        coefficient = c[i - 1]
        coefficient *= a[sum(_dim) + j - 1]
        try:
            coefficient *= lambda_matrix[np.sum(np.multiply(_dim, _deg)) + p]
        except IndexError:
            coefficient *= lambda_matrix[-1]
        return coefficient

    @classmethod
    def _show_dependence_by_polynomials(cls, a, c, lambda_matrix, dim, degrees, polynomial_type, dim_y):
        def _get_term(_i, _j, _p, __i):
            coefficient = cls.__get_coefficient(a[__i], c[__i], lambda_matrix[__i], dim, degrees, i, j, p)
            if _i == 1 and _j == 1 and _p == 0:
                sign = ""
            elif coefficient >= 0:
                sign = ""
            else:
                sign = "-"
            coefficient = abs(coefficient)
            return f"(1 + φ{_p}(x{_i}{_j}))^{sign}{coefficient:.3f} * "

        out = ""
        for _i in (dim_y,):
            out += f"Ф{_i+1} (x1, x2, x3) = "
            for i in np.arange(1, 4):  # i = 1 ... 3
                for j in np.arange(1, dim[i-1]+1):  # j = 1 ... dim[d]
                    for p in np.arange(degrees[i-1]+1):  # p = 0 ... degrees[i]
                        out += _get_term(i, j, p, _i)
            out = out[:-3]
            out += "\n\n"
        return out

    @classmethod
    def _show_special(cls, solver, y_i):
        psi = solver.psi
        a = solver.a
        c = solver.c
        out = cls._show_lambda(solver, y_i)
        out += cls._show_psi(psi, y_i)
        out += cls._show_a(a, y_i)
        out += cls._show_c(c, y_i)
        return out

    @classmethod
    def show(cls, solver):
        out = ""
        for dim in range(solver.dim_y):
            out += f"Для Ф{dim+1} (x1, x2, x3):\n\n"
            out += cls._show_special(solver, dim)
        out += "Відновлені через поліноми функції:\n\n"
        dim = np.array((solver.dim_x1, solver.dim_x2, solver.dim_x3))
        degrees = np.array((solver.x1_degree, solver.x2_degree, solver.x3_degree))
        for y_i in range(solver.dim_y):
            out += cls._show_dependence_by_polynomials(solver.a, solver.c, solver.lambda_matrix, dim, degrees, ..., y_i)
        out += "Нев'язка:\n"
        for y_i in range(solver.dim_y):
            out += f"Ф{y_i+1}: {solver.error_normalized[y_i]}\n"
        return out

    @classmethod
    def save(cls, solver):
        out = cls.show(solver)
        with open(solver.output, 'w', encoding='utf-16') as fileout:
            fileout.write(out)


