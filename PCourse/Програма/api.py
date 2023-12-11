class Options:
    def __init__(self, ui):
        self.input = self.__input(ui)
        self.output = self.__output(ui)
        self.sample_size = self.__sample_size(ui)
        self.dim_x1 = self.__dim_x1(ui)
        self.dim_x2 = self.__dim_x2(ui)
        self.dim_x3 = self.__dim_x3(ui)
        self.dim_y = self.__dim_y(ui)
        self.polynomial = self.__polynom(ui)
        self.x1_degree = self.__x1_degree(ui)
        self.x2_degree = self.__x2_degree(ui)
        self.x3_degree = self.__x3_degree(ui)
        self.weights = self.__weights(ui)
        self.lambda_options = self.__lambda_options(ui)
        self.own_function = ui.use_own_functional_structure.isChecked()

    def __input(self, ui):
        _input = ui.input_filename.text()
        return _input

    def __output(self, ui):
        _output = ui.output_filename.text()
        if _output == "":
            _output = "default.txt"
            ui.output_filename.setText(_output)
        return _output

    def __sample_size(self, ui):
        return ui.sample_size.value()

    def __dim_x1(self, ui):
        return ui.vector_x1.value()

    def __dim_x2(self, ui):
        return ui.vector_x2.value()

    def __dim_x3(self, ui):
        return ui.vector_x3.value()

    def __dim_y(self, ui):
        return ui.vector_y.value()

    def __polynom(self, ui):
        if ui.sh_legendre.isChecked():
            return "P*"
        elif ui.double_sh_legendre.isChecked():
            return "2P*"
        elif ui.legendre.isChecked():
            return "P"
        else:
            raise KeyError("Unknown polynomial type from UI!")

    def __x1_degree(self, ui):
        return ui.x_1_degree.value()

    def __x2_degree(self, ui):
        return ui.x_2_degree.value()

    def __x3_degree(self, ui):
        return ui.x_3_degree.value()

    def __weights(self, ui):
        if ui.b_normalized.isChecked():
            return "normalized"
        elif ui.b_average.isChecked():
            return "average"

    def __lambda_options(self, ui):
        if ui.lambda_one_system.isChecked():
            return False
        elif ui.lambda_three_systems.isChecked():
            return True
        else:
            raise KeyError("Unknown lambda option from UI!")
