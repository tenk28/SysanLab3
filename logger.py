import numpy as np


class Logger:
    @classmethod
    def get_log(cls, model):
        intermediate_values = ''
        F_first_forms = ''
        F_second_forms = ''
        residuals = ''
        for i in range(model.Y_dim):
            intermediate_value, F_first_form, F_second_form, residual = cls.get_values(model, i)

            intermediate_values += intermediate_value
            F_first_forms += F_first_form
            F_second_forms += F_second_form
            residuals += residual

        log = intermediate_values + \
              f'Відновлені в мультиплікативному вигляді функції:\n{F_first_forms}\n' + F_second_forms + \
              f"\nНев'язка:\n{residuals}"
        with open(model.output_filename, 'w', encoding='utf-8') as output_file:
            output_file.write(log)
        return log

    @classmethod
    def get_values(cls, model, i):
        inter_vals = cls.get_intermediate_values(model, i)
        F1 = cls.get_F_first_form(model, i)
        F2 = cls.get_F_second_form(model.a[i], model.c[i], model.lambdas[i], model.X_dims, model.P_dims)
        res = cls.get_residual(model, i)
        return inter_vals, F1, F2, res

    @classmethod
    def get_intermediate_values(cls, model, i):
        values = f'Для Ф{i + 1}:\n'
        values += f'Лямбда:\n{model.lambdas[i]}\n\n'
        values += f'Псі:\n{np.vstack(model.psi[i])}\n\n'
        values += f'А:\n{model.a[i]}\n\n'
        # values += f'{model.phi[i]}'
        values += f'С:\n{model.c[i]}\n\n\n'
        return values

    @classmethod
    def get_F_first_form(cls, model, i):
        return f'Ф{i + 1}(X1,X2,X3)=(1+Ф{i + 1}1(X1))^{model.c[i][0]:.6f}(1+Ф{i + 1}2(X2))^{model.c[i][1]:.6f}' \
               f'(1+Ф{i + 1}3(X3))^{model.c[i][2]:.6f}-1\n'

    @classmethod
    def get_F_second_form(cls, a, c, lambdas, X_dims, P_dims):
        def get_term(_i, _j, _p):
            coefficient = cls.get_coefficient(a, c, lambdas, X_dims, P_dims, _i - 1, _j - 1, _p)
            return f"(1+φ{_p + 1}(x{_i}{_j}))^{coefficient:.6f}"

        function = "Ф(X1,X2,X3)="
        for i in np.arange(1, 4):
            for j in np.arange(1, X_dims[i - 1] + 1):
                for p in np.arange(P_dims[i - 1]):
                    function += get_term(i, j, p)
        function += "-1\n"
        return function

    @staticmethod
    def get_coefficient(a, c, lambdas, X_dims, P_dims, i, j, p):
        _dim = X_dims[:i - 1]
        _deg = P_dims[:i - 1]
        coefficient = c[i - 1]
        coefficient *= a[np.sum(_dim) + j - 1]
        index_of_lambda = np.multiply(_dim, _deg)
        index_of_lambda = np.sum(index_of_lambda) + p - 1
        index_of_lambda = min(index_of_lambda, lambdas.size - 1)
        coefficient *= lambdas[index_of_lambda]
        return coefficient

    @classmethod
    def get_residual(cls, model, i):
        return f'Ф{i + 1}: {model.normalized_residuals[i]}\n'
