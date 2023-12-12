import numpy as np
from matplotlib import pyplot as plt


class Plot:
    @staticmethod
    def get_plot(model):
        samples = np.arange(1, model.Q + 1)
        number_of_graphs = model.residuals.size
        if number_of_graphs % 2 == 0:
            fig, axes = plt.subplots(2, number_of_graphs // 2)
            i_max = 2
            j_max = number_of_graphs // 2
        elif number_of_graphs == 1:
            fig, axes = plt.subplots(1, 1)
            i_max = 1
            j_max = 1
        else:
            fig, axes = plt.subplots(2, number_of_graphs // 2 + 1)
            i_max = 2
            j_max = number_of_graphs // 2 + 1

        def axe_ij(i, j):
            if number_of_graphs == 1:
                return axes
            elif number_of_graphs == 2:
                return axes[i]
            else:
                return axes[i][j]

        fig.subplots_adjust(hspace=0.3)
        for i in np.arange(i_max):
            for j in np.arange(j_max):
                k = i * j_max + j
                if k < number_of_graphs:
                    axe = axe_ij(i, j)
                    axe.plot(samples, model.Y_normalized[k], label=f'$Y_{k + 1}$')
                    axe.plot(samples, model.normalized_results[k], label=f'$Ф_{k + 1}$')
                    axe.set_title(f"Нев'язка: {model.normalized_residuals[k]:.6f}")
                    axe.legend()
                    axe.grid()
        fig.suptitle('Порівняння значень вибірки і апроксимованих')
        fig.savefig('Нормовані_графіки.png')
