from tkinter import *
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)

from logger import Logger
from model import *
from plot import Plot

np.set_printoptions(linewidth=np.inf)

window = Tk()
window.title('Лабораторна робота 3')
window.resizable(False, False)


class Application:
    def __init__(self, window):
        self.samples_label_frame = LabelFrame(window, text='Вхідні дані')
        self.samples_label_frame.grid(row=0, column=0, sticky='NW', padx=5, pady=5, ipadx=5, ipady=5)

        self.dimensions_label_frame = LabelFrame(window, text='Розмірності векторів')
        self.dimensions_label_frame.grid(row=1, column=0, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

        self.polynomials_label_frame = LabelFrame(window, text='Поліноми')
        self.polynomials_label_frame.grid(row=0, column=1, rowspan=2, sticky='NS', padx=5, pady=5, ipadx=5, ipady=5)

        self.polynomials_type_label_frame = LabelFrame(self.polynomials_label_frame, text='Вид поліномів')
        self.polynomials_type_label_frame.grid(row=0, column=0, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

        self.polynomials_dimensions_label_frame = LabelFrame(self.polynomials_label_frame, text='Степені поліномів')
        self.polynomials_dimensions_label_frame.grid(row=1, column=0, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

        self.additional_label_frame = LabelFrame(window, text='Додатково')
        self.additional_label_frame.grid(row=0, column=2, rowspan=2, sticky='N', padx=5, pady=5, ipadx=5, ipady=5)

        self.weight_label_frame = LabelFrame(self.additional_label_frame, text='Ваги цільових функцій')
        self.weight_label_frame.grid(row=0, column=0, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

        self.lambdas_label_frame = LabelFrame(self.additional_label_frame, text='Метод визначення лямбд')
        self.lambdas_label_frame.grid(row=1, column=0, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

        self.results_label_frame = LabelFrame(window, text='Результати')
        self.results_label_frame.grid(row=2, column=0, columnspan=3, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

        # 'Вхідні дані'
        self.samples_label = Label(self.samples_label_frame, text='Файл з вибіркою:')
        self.samples_label.grid(row=0, column=0, sticky='E', padx=5, pady=2)

        self.samples_filename_var = StringVar()
        self.samples_filename_var.set('')
        self.samples_filename_entry = Entry(self.samples_label_frame, textvariable=self.samples_filename_var,
                                            state=DISABLED)
        self.samples_filename_entry.grid(row=0, column=1, sticky='WE', padx=5, pady=2)

        self.browse_button = Button(self.samples_label_frame, text='...', command=self.browse_file)
        self.browse_button.grid(row=0, column=2, sticky='W', padx=5, pady=2)

        self.output_filename_label = Label(self.samples_label_frame, text='Вихідний файл:')
        self.output_filename_label.grid(row=1, column=0, sticky='E', padx=5, pady=2)

        self.output_filename_var = StringVar()
        self.output_filename_var.set('')
        self.output_filename_entry = Entry(self.samples_label_frame, textvariable=self.output_filename_var)
        self.output_filename_entry.grid(row=1, column=1)

        self.q_label = Label(self.samples_label_frame, text='Розмір вибірки:')
        self.q_label.grid(row=2, column=0, sticky='E', padx=5, pady=2)

        self.q_var = StringVar()
        self.q_var.set('')
        self.q_entry = Entry(self.samples_label_frame, textvariable=self.q_var, state=DISABLED)
        self.q_entry.grid(row=2, column=1)

        # 'Розмірності векторів'
        self.dimensions_label_frame.columnconfigure(1, weight=1)
        self.x1_label = Label(self.dimensions_label_frame, text='Розмірність X1:')
        self.x1_label.grid(row=0, column=0, sticky='E', padx=5, pady=2)
        self.x1_var = StringVar()
        self.x1_var.set('')
        self.x1_entry = Entry(self.dimensions_label_frame, textvariable=self.x1_var, state=DISABLED)
        self.x1_entry.grid(row=0, column=1, sticky='WE', padx=5, pady=2)

        self.x2_label = Label(self.dimensions_label_frame, text='Розмірність X2:')
        self.x2_label.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        self.x2_var = StringVar()
        self.x2_var.set('')
        self.x2_spinbox = Entry(self.dimensions_label_frame, textvariable=self.x2_var, state=DISABLED)
        self.x2_spinbox.grid(row=1, column=1, sticky='WE', padx=5, pady=2)

        self.x3_label = Label(self.dimensions_label_frame, text='Розмірність X3:')
        self.x3_label.grid(row=2, column=0, sticky='E', padx=5, pady=2)
        self.x3_var = StringVar()
        self.x3_var.set('')
        self.x3_entry = Entry(self.dimensions_label_frame, textvariable=self.x3_var, state=DISABLED)
        self.x3_entry.grid(row=2, column=1, sticky='WE', padx=5, pady=2)

        self.y_label = Label(self.dimensions_label_frame, text='Розмірність Y:')
        self.y_label.grid(row=3, column=0, sticky='E', padx=5, pady=2)
        self.y_var = StringVar()
        self.y_var.set('')
        self.y_entry = Entry(self.dimensions_label_frame, textvariable=self.y_var, state=DISABLED)
        self.y_entry.grid(row=3, column=1, sticky='WE', padx=5, pady=2)

        # 'Поліноми'
        # 'Вид поліномів'
        self.polynomial_var = StringVar()
        self.polynomial_var.set(PolynomialMethod.SHIFTED_LEGANDRE.name)
        self.chebyshev_radiobutton = Radiobutton(self.polynomials_type_label_frame, text='P*(x)',
                                                 variable=self.polynomial_var,
                                                 value=PolynomialMethod.SHIFTED_LEGANDRE.name)
        self.chebyshev_radiobutton.grid(row=0, sticky='W')
        self.legandre_radiobutton = Radiobutton(self.polynomials_type_label_frame, text='2P*(x)',
                                                variable=self.polynomial_var,
                                                value=PolynomialMethod.DOUBLED_SHIFTED_LEGANDRE.name)
        self.legandre_radiobutton.grid(row=1, sticky='W')
        self.lagerr_radiobutton = Radiobutton(self.polynomials_type_label_frame, text='P(x)',
                                              variable=self.polynomial_var,
                                              value=PolynomialMethod.LEGANDRE.name)
        self.lagerr_radiobutton.grid(row=2, sticky='W')

        # 'Степені поліномів'
        self.polynomials_dimensions_label_frame.columnconfigure(1, weight=1)
        self.p1_label = Label(self.polynomials_dimensions_label_frame, text='P1:')
        self.p1_label.grid(row=0, column=0, sticky='E')
        self.p1_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5)
        self.p1_spinbox.grid(row=0, column=1, sticky='WE', padx=5, pady=2)

        self.p2_label = Label(self.polynomials_dimensions_label_frame, text='P2:')
        self.p2_label.grid(row=1, column=0, sticky='E')
        self.p2_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5)
        self.p2_spinbox.grid(row=1, column=1, sticky='WE', padx=5, pady=2)

        self.p3_label = Label(self.polynomials_dimensions_label_frame, text='P3:')
        self.p3_label.grid(row=2, column=0, sticky='E')
        self.p3_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5)
        self.p3_spinbox.grid(row=2, column=1, sticky='WE', padx=5, pady=2)

        # 'Додатково'
        # 'Ваги цільових функцій'
        self.weight = StringVar()
        self.weight.set(WeightMethod.NORMED.name)
        self.normed_radiobutton = Radiobutton(self.weight_label_frame, text='Нормовані Yi', variable=self.weight,
                                              value=WeightMethod.NORMED.name)
        self.normed_radiobutton.grid(row=0, sticky='W')
        self.min_max_radiobutton = Radiobutton(self.weight_label_frame, text='(min(Yi) + max(Yi)) / 2',
                                               variable=self.weight,
                                               value=WeightMethod.MIN_MAX.name)
        self.min_max_radiobutton.grid(row=1, sticky='W')

        # 'Метод визначення лямбд'
        self.lambdas = StringVar()
        self.lambdas.set(LambdaMethod.SINGLE_SET.name)
        self.single_set_radiobutton = Radiobutton(self.lambdas_label_frame, text='Одна система', variable=self.lambdas,
                                                  value=LambdaMethod.SINGLE_SET.name)
        self.single_set_radiobutton.grid(row=0, sticky='W')
        self.triple_set_radiobutton = Radiobutton(self.lambdas_label_frame, text='Три системи', variable=self.lambdas,
                                                  value=LambdaMethod.TRIPLE_SET.name)
        self.triple_set_radiobutton.grid(row=1, sticky='W')

        # 'Результати'
        self.result_area = ScrolledText(self.results_label_frame, height=10)
        self.result_area.grid(row=0, column=0, sticky='WENS')

        self.calculate_button = Button(self.additional_label_frame, text='Обрахувати',
                                       command=self.calculate,
                                       bg='red',
                                       fg='white'
                                       )
        self.calculate_button.grid(sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

    def browse_file(self):
        samples_filename = filedialog.askopenfilename(title='Open a File',
                                                      filetypes=(("Excel files", ".*xlsx"), ("All Files", "*.")))
        self.samples_filename_var.set(samples_filename)
        return samples_filename

    def get_entries(self):
        samples_filename = None
        output_filename = 'default'

        if self.samples_filename_entry.get() != '':
            samples_filename = self.samples_filename_entry.get()

        if self.output_filename_entry.get() != '':
            output_filename = self.output_filename_entry.get()

        P_dims = [int(self.p1_spinbox.get()), int(self.p2_spinbox.get()), int(self.p3_spinbox.get())]
        return samples_filename, output_filename, P_dims, self.polynomial_var.get(), self.weight.get(), self.lambdas.get(), False

    def calculate(self):
        model = Model(*self.get_entries())
        log = Logger.get_log(model)
        plot = Plot.get_plot(model)

        self.q_var.set(str(model.Q))
        self.x1_var.set(str(model.X1_dim))
        self.x2_var.set(str(model.X2_dim))
        self.x3_var.set(str(model.X3_dim))
        self.y_var.set(str(model.Y_dim))

        self.result_area.insert(END, log)


application = Application(window)

window.mainloop()
