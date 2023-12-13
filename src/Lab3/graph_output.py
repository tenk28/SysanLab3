import matplotlib.pyplot as plt
import numpy as np
from solve import Solve
import matplotlib
matplotlib.use('qtagg')

class Graph:
    def __init__(self, ui, normalized=False):
        solver = Solve(ui)
        self.plot_normalized = self._is_normalized(normalized)
        self.estimate = self._get_estimate(solver)
        self.error = self._get_error(solver)
        self.y = self._get_y(solver)
        self.sample_size = solver.sample_size

    def _get_y(self, solver):
        if self.plot_normalized:
            return solver.y_normalized
        else:
            return solver.y

    def _get_title(self):
        if self.plot_normalized:
            title = "Порівняння значень вибірки і апроксимованих"
        else:
            title = "Порівняння значень оригінальної вибірки і відновлених апроксимованих"
        return title

    @staticmethod
    def _is_normalized(plot_normalized):
        return plot_normalized

    def _get_estimate(self, solver):
        if self.plot_normalized:
            return solver.estimate_normalized
        else:
            return solver.estimate
            
    def _get_error(self, solver):
        if self.plot_normalized:
            return solver.error_normalized
        else:
            return solver.error

    def plot_graph(self):
        samples = np.arange(1, self.sample_size+1)
        number_of_graphs = self.error.size
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

        def _axe(i, j):
            if number_of_graphs == 1:
                return axes
            elif number_of_graphs == 2:
                return axes[i]
            else:
                return axes[i][j]

        for i in np.arange(i_max):
            for j in np.arange(j_max):
                k = i * j_max + j
                if k < number_of_graphs:
                    axe = _axe(i, j)
                    axe.plot(samples, self.y[k], label=f'$Y_{k + 1}$')
                    axe.plot(samples, self.estimate[k], label=f'$F_{k + 1}$')
                    axe.set_title(f"Нев'язка: {self.error[k]:.6f}")
                    axe.legend()
                    axe.grid()
        title = self._get_title()
        fig.suptitle(title)
        fig.show()
