"""
This is the implementation of the prediction application in the project. Select the appropriate features and make
prediction of default or non-default on credit card.
Created on Nov.29, 2017

@author Ted
"""

from tkinter import Tk, Label, Button, Frame, BOTTOM, LEFT, TOP, RIGHT, StringVar
from tkinter.ttk import Combobox
from sklearn.svm import SVC
from Preprocessing import Preprocess
from Postprocessing import Postprocess
import numpy as np


class predictor:
    """The class of predictor"""

    def __init__(self):
        self.c = SVC()
        self.trainmodel()

    # train the model of making prediction, we load data with reduced dimensions and discretization as training set
    def trainmodel(self):
        prep = Preprocess("default of credit card clients.xls")
        prep.load()
        low_dim_x1, low_dim_x2, low_dim_y1, low_dim_y2 = prep.dimension_decrease()
        postp = Postprocess(low_dim_x1, low_dim_x2, low_dim_y1, low_dim_y2)
        discretized_x1, discretized_x2, discretized_y1, discretized_y2 = postp.improve_data()
        x = np.concatenate((discretized_x1, discretized_x2))
        y = np.concatenate((discretized_y1, discretized_y2))
        self.c.fit(x, y)

        y_pred = self.c.predict(x)
        mislabeled = (y != y_pred).sum()
        totaltest = x.shape[0]
        print("Mislabeled points (%s Classification) out of a total %d points : %d" % ("SVC", totaltest, mislabeled))
        Precision = 1 - mislabeled / totaltest
        print("Precision of %s is %4.2f%%" % ("SVC", Precision * 100))


class MyPredictionGUI:
    """The class of application GUI"""

    # initial the grid and layout of the application
    def __init__(self, master):
        self.predictor = predictor()

        self.master = master
        master.title("A simple GUI")
        master.geometry('355x355')

        self.main_label = Label(master, text="Simple Credit Card Default Evaluation")
        self.main_label.pack()

        self.topFrame = Frame(master)
        self.topFrame.pack()

        self.botFrame = Frame(master)
        self.botFrame.pack(side=BOTTOM)

        self.input_label = []
        self.input_label_text = ["Balance Limit", "Gender", "Education", "Marriage", "Age", "Missed payment",
                                 "Amount Owed"]
        self.input_component = []
        self.input_component_text = [["None", "1 - 100,000", "100,001 - 500,000", "500,001 and more"],
                                     ["Male", "Female"],
                                     ["Graduate school", "University", "High school", "Others"],
                                     ["Married", "Single", "Others"],
                                     ["1-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80",
                                      "80 and more"],
                                     ["None", "1", "2", "3", "4 and more"],
                                     ["None", "Low", "Medium", "High"]]

        self.input_value = []

        for i in range(0, 7):
            temp_label = Label(self.topFrame, text=self.input_label_text[i], width=15)
            self.input_label.append(temp_label)
            temp_label.grid(row=i, column=0)

            temp_value = StringVar()
            self.input_value.append(temp_value)
            temp_input_component = Combobox(self.topFrame, textvariable=temp_value, width=15)
            self.input_component.append(temp_input_component)
            temp_input_component.grid(row=i, column=1)
            temp_input_component["values"] = self.input_component_text[i]
            temp_input_component.current(0)

        self.prediction_text = 'Select each group you belong to'
        self.prediction_label = Label(self.botFrame, text=self.prediction_text)
        self.prediction_label.pack()

        self.greet_button = Button(self.botFrame, text="Predict", command=self.predict)
        self.greet_button.pack(side=LEFT)

        self.close_button = Button(self.botFrame, text="Close", command=master.quit)
        self.close_button.pack(side=RIGHT)

    # handle the selected input
    def modify_input(self):
        result = []
        for i in range(0, 7):
            if i >= 5 or i == 0:
                result.append(self.input_component_text[i].index(self.input_value[i].get()))
            else:
                result.append(self.input_component_text[i].index(self.input_value[i].get())+1)
        return result

    # predict and show the result
    def predict(self):
        modified_input = np.asarray(self.modify_input()).reshape(1, -1)
        prediction = self.predictor.c.predict(modified_input)
        if prediction == 0:
            self.prediction_label["text"] = "The result is no Default"
        else:
            self.prediction_label["text"] = "The result is Default"
        print("prediction value is: %d" % prediction)


if __name__ == "__main__":
    root = Tk()
    my_gui = MyPredictionGUI(root)
    root.mainloop()