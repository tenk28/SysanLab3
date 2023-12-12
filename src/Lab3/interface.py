# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(691, 703)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setMaximumSize(QtCore.QSize(1264, 782))
        MainWindow.setAcceptDrops(False)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color: rgb(254, 205, 166);")
        self.centralwidget.setObjectName("centralwidget")
        self.input_data_text = QtWidgets.QGroupBox(self.centralwidget)
        self.input_data_text.setGeometry(QtCore.QRect(700, 10, 301, 121))
        self.input_data_text.setStyleSheet("")
        self.input_data_text.setObjectName("input_data_text")
        self.input_data_text.setStyleSheet("background-color: rgb(236, 227, 206);")
        self.widget = QtWidgets.QWidget(self.input_data_text)
        self.widget.setGeometry(QtCore.QRect(10, 20, 282, 85))
        self.widget.setObjectName("widget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.input_filename = QtWidgets.QLineEdit(self.widget)
        self.input_filename.setText("")
        self.input_filename.setDragEnabled(False)
        self.input_filename.setPlaceholderText("")
        self.input_filename.setObjectName("input_filename")
        self.gridLayout_2.addWidget(self.input_filename, 0, 1, 1, 1)
        self.input_choose = QtWidgets.QToolButton(self.widget)
        self.input_choose.setStyleSheet("background-color: rgb(150, 150, 150);")
        self.input_choose.setObjectName("input_choose")
        self.gridLayout_2.addWidget(self.input_choose, 0, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 1, 0, 1, 1)
        self.output_filename = QtWidgets.QLineEdit(self.widget)
        self.output_filename.setText("")
        self.output_filename.setPlaceholderText("")
        self.output_filename.setObjectName("output_filename")
        self.gridLayout_2.addWidget(self.output_filename, 1, 1, 1, 1)
        self.sample_size_txt = QtWidgets.QLabel(self.widget)
        self.sample_size_txt.setObjectName("sample_size_txt")
        self.gridLayout_2.addWidget(self.sample_size_txt, 2, 0, 1, 1)
        self.sample_size = QtWidgets.QSpinBox(self.widget)
        self.sample_size.setMouseTracking(False)
        self.sample_size.setMinimum(1)
        self.sample_size.setMaximum(100)
        self.sample_size.setObjectName("sample_size")
        self.gridLayout_2.addWidget(self.sample_size, 2, 1, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(700, 140, 301, 141))
        self.groupBox_2.setStyleSheet("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_2.setStyleSheet("background-color: rgb(236, 227, 206);")
        self.widget1 = QtWidgets.QWidget(self.groupBox_2)
        self.widget1.setGeometry(QtCore.QRect(10, 25, 281, 111))
        self.widget1.setObjectName("widget1")
        self.gridLayout = QtWidgets.QGridLayout(self.widget1)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.vector_x1_text = QtWidgets.QLabel(self.widget1)
        self.vector_x1_text.setObjectName("vector_x1_text")
        self.gridLayout.addWidget(self.vector_x1_text, 0, 0, 1, 1)
        self.vector_x1 = QtWidgets.QSpinBox(self.widget1)
        self.vector_x1.setMouseTracking(False)
        self.vector_x1.setMinimum(1)
        self.vector_x1.setObjectName("vector_x1")
        self.gridLayout.addWidget(self.vector_x1, 0, 1, 1, 1)
        self.vector_x2_text = QtWidgets.QLabel(self.widget1)
        self.vector_x2_text.setObjectName("vector_x2_text")
        self.gridLayout.addWidget(self.vector_x2_text, 1, 0, 1, 1)
        self.vector_x2 = QtWidgets.QSpinBox(self.widget1)
        self.vector_x2.setMinimum(1)
        self.vector_x2.setObjectName("vector_x2")
        self.gridLayout.addWidget(self.vector_x2, 1, 1, 1, 1)
        self.vector_x3_text = QtWidgets.QLabel(self.widget1)
        self.vector_x3_text.setObjectName("vector_x3_text")
        self.gridLayout.addWidget(self.vector_x3_text, 2, 0, 1, 1)
        self.vector_x3 = QtWidgets.QSpinBox(self.widget1)
        self.vector_x3.setMinimum(1)
        self.vector_x3.setObjectName("vector_x3")
        self.gridLayout.addWidget(self.vector_x3, 2, 1, 1, 1)
        self.vector_y_text = QtWidgets.QLabel(self.widget1)
        self.vector_y_text.setObjectName("vector_y_text")
        self.gridLayout.addWidget(self.vector_y_text, 3, 0, 1, 1)
        self.vector_y = QtWidgets.QSpinBox(self.widget1)
        self.vector_y.setMinimum(1)
        self.vector_y.setObjectName("vector_y")
        self.gridLayout.addWidget(self.vector_y, 3, 1, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(700, 290, 301, 140))
        self.groupBox_3.setObjectName("groupBox_3")
        self.groupBox_3.setStyleSheet("background-color: rgb(236, 227, 206);")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 25, 131, 111))
        self.groupBox_4.setObjectName("groupBox_4")
        self.groupBox_4.setStyleSheet("background-color: rgb(236, 227, 206);")
        self.widget2 = QtWidgets.QWidget(self.groupBox_4)
        self.widget2.setGeometry(QtCore.QRect(10, 25, 111, 76))
        self.widget2.setObjectName("widget2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget2)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.sh_legendre = QtWidgets.QRadioButton(self.widget2)
        self.sh_legendre.setChecked(True)
        self.sh_legendre.setObjectName("sh_legendre")
        self.gridLayout_3.addWidget(self.sh_legendre, 0, 0, 1, 1)
        self.double_sh_legendre = QtWidgets.QRadioButton(self.widget2)
        self.double_sh_legendre.setObjectName("double_sh_legendre")
        self.gridLayout_3.addWidget(self.double_sh_legendre, 1, 0, 1, 1)
        self.legendre = QtWidgets.QRadioButton(self.widget2)
        self.legendre.setObjectName("legendre")
        self.gridLayout_3.addWidget(self.legendre, 2, 0, 1, 1)
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_5.setGeometry(QtCore.QRect(145, 25, 145, 111))
        self.groupBox_5.setObjectName("groupBox_5")
        self.groupBox_5.setStyleSheet("background-color: rgb(236, 227, 206);")
        self.widget3 = QtWidgets.QWidget(self.groupBox_5)
        self.widget3.setGeometry(QtCore.QRect(18, 25, 101, 82))
        self.widget3.setObjectName("widget3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget3)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.polynom_degree_x1_text_2 = QtWidgets.QLabel(self.widget3)
        self.polynom_degree_x1_text_2.setObjectName("polynom_degree_x1_text_2")
        self.gridLayout_4.addWidget(self.polynom_degree_x1_text_2, 0, 0, 1, 1)
        self.x_1_degree = QtWidgets.QSpinBox(self.widget3)
        self.x_1_degree.setMouseTracking(False)
        self.x_1_degree.setMinimum(1)
        self.x_1_degree.setObjectName("x_1_degree")
        self.gridLayout_4.addWidget(self.x_1_degree, 0, 1, 1, 1)
        self.polynom_degree_x1_text = QtWidgets.QLabel(self.widget3)
        self.polynom_degree_x1_text.setObjectName("polynom_degree_x1_text")
        self.gridLayout_4.addWidget(self.polynom_degree_x1_text, 1, 0, 1, 1)
        self.x_2_degree = QtWidgets.QSpinBox(self.widget3)
        self.x_2_degree.setMinimum(1)
        self.x_2_degree.setObjectName("x_2_degree")
        self.gridLayout_4.addWidget(self.x_2_degree, 1, 1, 1, 1)
        self.polynom_degree_x3_text = QtWidgets.QLabel(self.widget3)
        self.polynom_degree_x3_text.setObjectName("polynom_degree_x3_text")
        self.gridLayout_4.addWidget(self.polynom_degree_x3_text, 2, 0, 1, 1)
        self.x_3_degree = QtWidgets.QSpinBox(self.widget3)
        self.x_3_degree.setMinimum(1)
        self.x_3_degree.setObjectName("x_3_degree")
        self.gridLayout_4.addWidget(self.x_3_degree, 2, 1, 1, 1)
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(700, 440, 301, 261))
        self.groupBox_6.setObjectName("groupBox_6")
        self.groupBox_6.setStyleSheet("background-color: rgb(236, 227, 206);")
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_7.setGeometry(QtCore.QRect(10, 25, 280, 81))
        self.groupBox_7.setObjectName("groupBox_7")
        self.groupBox_7.setStyleSheet("background-color: rgb(236, 227, 206);")
        self.widget4 = QtWidgets.QWidget(self.groupBox_7)
        self.widget4.setGeometry(QtCore.QRect(10, 25, 160, 49))
        self.widget4.setObjectName("widget4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.widget4)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.b_normalized = QtWidgets.QRadioButton(self.widget4)
        self.b_normalized.setChecked(True)
        self.b_normalized.setObjectName("b_normalized")
        self.gridLayout_5.addWidget(self.b_normalized, 0, 0, 1, 1)
        self.b_average = QtWidgets.QRadioButton(self.widget4)
        self.b_average.setObjectName("b_average")
        self.gridLayout_5.addWidget(self.b_average, 1, 0, 1, 1)
        self.groupBox_8 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_8.setGeometry(QtCore.QRect(10, 109, 280, 81))
        self.groupBox_8.setObjectName("groupBox_8")
        self.widget5 = QtWidgets.QWidget(self.groupBox_8)
        self.widget5.setGeometry(QtCore.QRect(9, 25, 161, 49))
        self.widget5.setObjectName("widget5")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.widget5)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.lambda_one_system = QtWidgets.QRadioButton(self.widget5)
        self.lambda_one_system.setChecked(True)
        self.lambda_one_system.setObjectName("lambda_one_system")
        self.gridLayout_6.addWidget(self.lambda_one_system, 0, 0, 1, 1)
        self.lambda_three_systems = QtWidgets.QRadioButton(self.widget5)
        self.lambda_three_systems.setObjectName("lambda_three_systems")
        self.gridLayout_6.addWidget(self.lambda_three_systems, 1, 0, 1, 1)
        self.use_own_functional_structure = QtWidgets.QCheckBox(self.groupBox_6)
        self.use_own_functional_structure.setGeometry(QtCore.QRect(10, 200, 179, 20))
        self.use_own_functional_structure.setObjectName("use_own_functional_structure")
        self.execute_button = QtWidgets.QPushButton(self.groupBox_6)
        self.execute_button.setGeometry(QtCore.QRect(65, 230, 181, 21))
        self.execute_button.setStyleSheet("background-color: rgb(166, 207, 152);")
        self.execute_button.setObjectName("execute_button")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 671, 690))
        self.groupBox.setObjectName("groupBox")
        self.groupBox.setStyleSheet("background-color: rgb(236, 227, 206);")
        self.output_field = QtWidgets.QTextBrowser(self.groupBox)
        self.output_field.setGeometry(QtCore.QRect(10, 20, 651, 660))
        self.output_field.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.output_field.setObjectName("output_field")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Бригада №6"))
        self.input_data_text.setTitle(_translate("MainWindow", "Вхідні дані"))
        self.label.setText(_translate("MainWindow", "Файл з вибіркою"))
        self.input_choose.setText(_translate("MainWindow", "..."))
        self.label_4.setText(_translate("MainWindow", "Вихідний файл"))
        self.sample_size_txt.setText(_translate("MainWindow", "Розмір вибірки"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Розмірності векторів"))
        self.vector_x1_text.setText(_translate("MainWindow", "Розмірність Х1:"))
        self.vector_x2_text.setText(_translate("MainWindow", "Розмірність Х2:"))
        self.vector_x3_text.setText(_translate("MainWindow", "Розмірність Х3:"))
        self.vector_y_text.setText(_translate("MainWindow", "Розмірність Y:"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Базові функції"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Вид базової функції"))
        self.sh_legendre.setText(_translate("MainWindow", "φ(x)=P*(x)"))
        self.double_sh_legendre.setText(_translate("MainWindow", "φ(x)=2P*(x)"))
        self.legendre.setText(_translate("MainWindow", "φ(x)=P(x)"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Степені поліномів"))
        self.polynom_degree_x1_text_2.setText(_translate("MainWindow", "P1"))
        self.polynom_degree_x1_text.setText(_translate("MainWindow", "P2"))
        self.polynom_degree_x3_text.setText(_translate("MainWindow", "P3"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Додатково"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Ваги цільових функцій"))
        self.b_normalized.setText(_translate("MainWindow", "Нормовані Yi"))
        self.b_average.setText(_translate("MainWindow", "(min(Yi) + max(Yi)) / 2"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Метод визначення лямбд"))
        self.lambda_one_system.setText(_translate("MainWindow", "Одна система"))
        self.lambda_three_systems.setText(_translate("MainWindow", "Три системи"))
        self.use_own_functional_structure.setText(_translate("MainWindow", "Власна структура функцій"))
        self.execute_button.setText(_translate("MainWindow", "Обрахувати"))
        self.groupBox.setTitle(_translate("MainWindow", "Результати"))
        self.output_field.setHtml(
            _translate(
                "MainWindow",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
                '<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;"><br /></p></body></html>',
            )
        )
