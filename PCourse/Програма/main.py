#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
from interface import Ui_MainWindow
from solve import Solve
from text_output import Output
from graph_output import Graph


class UI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.input_choose.clicked.connect(self.choose_input)
        self.execute_button.clicked.connect(self.execute)
        self.set_default_io()

    def choose_input(self):
        filename = QFileDialog.getOpenFileName(self, 'Open data file', '.', 'Data file (*.txt)')[0]
        self.input_filename.setText(filename)

    def choose_output(self):
        filename = QFileDialog.getOpenFileName(self, 'Open data file', '.', 'Data file (*.txt)')[0]
        self.output_filename.setText(filename)

    def set_default_io(self):
        default_input = 'ТестоваВибірка.txt'
        self.input_filename.setText(default_input)
        default_output = 'default.txt'
        self.output_filename.setText(default_output)
        self.vector_x1.setValue(2)
        self.vector_x2.setValue(2)
        self.vector_x3.setValue(3)
        self.vector_y.setValue(4)
        self.sample_size.setValue(45)
        self.x_1_degree.setValue(3)
        self.x_2_degree.setValue(3)
        self.x_3_degree.setValue(3)

    def execute(self):
        self.execute_button.setEnabled(False)
        solver = Solve(self)
        self.output_field.setText(Output.show(solver))
        Output.save(solver)
        plotter = Graph(self, False)
        plotter.plot_graph()
        plotter = Graph(self, True)
        plotter.plot_graph()
        self.execute_button.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = UI()
    MainWindow.show()

    sys.exit(app.exec_())