"""
Postprocess the data, read the preprocessed data and apply discretization on it. Set the attribute AGE into equal
interval of ten, and placed both attributes AMT_OWED and LIMIT_BAL into level ranged as “Low” for 0-100,000, “Medium”
for 100,000-500,000, and “High” for 500,000-more respectively.
Created on Nov 29, 2017

@author: Zac
"""

from Preprocessing import Preprocess


class Postprocess():
    def __init__(self, x1, x2, y1, y2):
        self.x_train = x1
        self.x_test = x2
        self.y_train = y1
        self.y_test = y2

    def set_age(self):
        # group age into different intervals
        for row in range(0, 10616):
            temp = self.x_train[row, 4]
            if 1 <= temp < 10:
                self.x_train[row, 4] = 1
            elif 10 <= temp < 20:
                self.x_train[row, 4] = 2
            elif 20 <= temp < 30:
                self.x_train[row, 4] = 3
            elif 30 <= temp < 40:
                self.x_train[row, 4] = 4
            elif 40 <= temp < 50:
                self.x_train[row, 4] = 5
            elif 50 <= temp < 60:
                self.x_train[row, 4] = 6
            elif 60 <= temp < 70:
                self.x_train[row, 4] = 7
            elif 70 <= temp < 80:
                self.x_train[row, 4] = 8
            else:
                self.x_train[row, 4] = 9

        for row in range(0, 2656):
            temp = self.x_test[row, 4]
            if 1 <= temp < 10:
                self.x_test[row, 4] = 1
            elif 10 <= temp < 20:
                self.x_test[row, 4] = 2
            elif 20 <= temp < 30:
                self.x_test[row, 4] = 3
            elif 30 <= temp < 40:
                self.x_test[row, 4] = 4
            elif 40 <= temp < 50:
                self.x_test[row, 4] = 5
            elif 50 <= temp < 60:
                self.x_test[row, 4] = 6
            elif 60 <= temp < 70:
                self.x_test[row, 4] = 7
            elif 70 <= temp < 80:
                self.x_test[row, 4] = 8
            else:
                self.x_test[row, 4] = 9

    def set_amount(self):

        # process training data
        for row in range(0, 10616):

            temp_lim = self.x_train[row, 0]
            temp_owe = self.x_train[row, 6]
            # group credit limit into different intervals
            if temp_lim == 0:
                continue
            elif 1 <= temp_lim < 100001:
                self.x_train[row, 0] = 1
            elif 100001 <= temp_lim < 500001:
                self.x_train[row, 0] = 2
            else:
                self.x_train[row, 0] = 3

            # group owed amount into different intervals
            if -100000 <= temp_owe < 0:
                self.x_train[row, 6] = -1
            elif -500000 <= temp_owe < -100000:
                self.x_train[row, 6] = -2
            elif temp_owe < -500000:
                self.x_train[row, 6] = -3
            elif self.x_train[row, 6] == 0:
                continue
            elif 1 <= temp_owe < 100001:
                self.x_train[row, 6] = 1
            elif 10000 <= temp_owe < 500001:
                self.x_train[row, 6] = 2
            else:
                self.x_train[row, 6] = 3

        # process testing data
        for row in range(0, 2656):

            temp_lim = self.x_test[row, 0]
            temp_owe = self.x_test[row, 6]
            # group credit limit into different intervals
            if temp_lim == 0:
                continue
            elif 1 <= temp_lim < 100001:
                self.x_test[row, 0] = 1
            elif 100001 <= temp_lim < 500001:
                self.x_test[row, 0] = 2
            else:
                self.x_test[row, 0] = 3

            # group owed amount into different intervals
            if -100000 <= temp_owe < 0:
                self.x_test[row, 6] = -1
            elif -500000 <= temp_owe < -100000:
                self.x_test[row, 6] = -2
            elif temp_owe < -500000:
                self.x_test[row, 6] = -3
            elif self.x_test[row, 6] == 0:
                continue
            elif 1 <= temp_owe < 100001:
                self.x_test[row, 6] = 1
            elif 10000 <= temp_owe < 500001:
                self.x_test[row, 6] = 2
            else:
                self.x_test[row, 6] = 3

    def improve_data(self):
        self.set_age()
        self.set_amount()
        return self.x_train, self.x_test, self.y_train, self.y_test


if __name__ == '__main__':
    a = Preprocess("default of credit card clients.xls")
    rx1, rx2, ry1, ry2 = a.load()
    x1, x2, y1, y2 = a.dimension_decrease()
    b = Postprocess(x1, x2, y1, y2)
    xd1, xd2, yd1, yd2 = b.improve_data()

