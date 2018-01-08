"""
This is the program of conducting the experiments designed in the project. It shows the comparisons between prediction
precisions of different classification techniques using raw data and different standardization techniques. Each row
displays one of the classification methods. And despite the column with classifier names, the first column represent
the precision made by raw data with 25 attributes. The other columns shows each of the standardization and normalization
techniques.

Created on Nov.25, 2017

@author Ted
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, Binarizer, QuantileTransformer
from Preprocessing import Preprocess
from Postprocessing import Postprocess


class experiment:

    # initialization function, load dataset and adjust preprocessing and postprocessing
    def __init__(self):
        self.classifier = []
        self.processor =[]
        self.result = []
        creditdata = Preprocess("default of credit card clients.xls")
        self.raw_X_train, self.raw_X_test, self.raw_Y_train, self.raw_Y_test = creditdata.load()
        self.low_dim_X_train, self.low_dim_X_test, self.low_dim_Y_train, self.low_dim_Y_test = \
            creditdata.dimension_decrease()
        x1, x2, y1, y2 = self.low_dim_X_train, self.low_dim_X_test, self.low_dim_Y_train, self.low_dim_Y_test
        self.discretizer = Postprocess(x1, x2, y1, y2)
        self.discretized_X_train, self.discretized_X_test, self.discretized_Y_train, self.discretized_Y_test = \
            self.discretizer.improve_data()
        self.buildclf()
        self.buildprocessor()
        self.logfile = open("execution_Log", "a")

    # function to set up a list of classifiers with different algorithm
    def buildclf(self):
        self.classifier.append(processor(GaussianNB(), "Gaussian NB"))
        self.classifier.append(processor(KNeighborsClassifier(n_neighbors=15), "K Nearest Neighbors"))
        self.classifier.append(processor(SVC(), "C-Support Vector"))
        self.classifier.append(processor(LogisticRegression(), "Logistic Regression"))
        self.classifier.append(processor(LinearDiscriminantAnalysis(), "Discriminant Analysis"))
        self.classifier.append(processor(MLPClassifier(), "Artificial neural networks"))
        self.classifier.append(processor(DecisionTreeClassifier(), "Decision Tree"))

    # function to set up a list of normalizers(scalers)
    def buildprocessor(self):
        self.processor.append(processor(StandardScaler(), "Standard Scaler"))
        self.processor.append(processor(MinMaxScaler(), "MinMax Scaler"))
        self.processor.append(processor(MaxAbsScaler(), "MaxAbs Scaler"))
        self.processor.append(processor(Normalizer(), "Normalization"))
        self.processor.append(processor(Binarizer(), "Binarization"))
        self.processor.append(processor(QuantileTransformer(), "Nonlinear transformation"))

    # calculate precision of given classifier and dataset
    def getprecision(self, clf, x1, x2, y1, y2):

        clf.obj.fit(x1, y1)
        y_pred = clf.obj.predict(x2)
        mislabeled = (y2 != y_pred).sum()
        totaltest = x2.shape[0]
        Precision = 1 - mislabeled / totaltest
        return Precision

    # calculate the precision and make the comparison table of the precisions with different classification and
    # normalization(standardization) method
    def compare_clf_and_prep(self, x1_raw, x2_raw, y1_raw, y2_raw, x1, x2, y1, y2):

        result_matrix = []
        maxprec = [0, "", ""]
        for clf in self.classifier:
            row = []
            prec = self.getprecision(clf, x1_raw, x2_raw, y1_raw, y2_raw)
            if prec > maxprec[0]:
                maxprec = [prec, "no preprocess", clf.descr]
            row.append(prec)
            for processor in self.processor:
                processed_X_train = processor.obj.fit_transform(x1)
                processed_X_test = processor.obj.fit_transform(x2)
                prec = self.getprecision(clf, processed_X_train, processed_X_test, y1, y2)
                if prec > maxprec[0]:
                    maxprec = [prec, processor.descr, clf.descr]
                row.append(prec)
            result_matrix.append(row)
        self.printresult(result_matrix)
        print("The maximum precision is %0.8f%%, with %s and %s classification" %(maxprec[0]*100, maxprec[1],
                                                                                 maxprec[2]))
        self.logfile.write("%0.8f%%, %s and %s classification" %(maxprec[0]*100, maxprec[1], maxprec[2]))
        self.logfile.write("\n")

    # print the result of given matrix(the table of comparison of precisions)
    def printresult(self, matrix):
        print("%26s%26s" %("_________________________", "No preprocess"), end='')
        for p in self.processor:
            print("%26s" %p.descr, end='')
        print('')
        for i in range(len(self.classifier)):
            print("%26s" % self.classifier[i].descr, end='')
            for j in range(len(self.processor)+1):
                print("%20s%4.2f%%" % ("", matrix[i][j]*100), end='')
            print('')

    # function to compare the performance of different dataset with same SVM classifier and non-linear normalizer
    def comparison(self, x1_raw, x2_raw, y1_raw, y2_raw, x1, x2, y1, y2, dx1, dx2, dy1, dy2):
        c = self.classifier[2]
        s = self.processor[5]

        preprocessed_X_train = s.obj.fit_transform(x1)
        preprocessed_X_test = s.obj.fit_transform(x2)
        preprocessed_dX_train = s.obj.fit_transform(dx1)
        preprocessed_dX_test = s.obj.fit_transform(dx2)

        comparison_result = [self.getprecision(c, x1_raw, x2_raw, y1_raw, y2_raw),
                             self.getprecision(c, preprocessed_X_train, preprocessed_X_test, y1, y2),
                             self.getprecision(c, preprocessed_dX_train, preprocessed_dX_test, dy1, dy2),
                             self.getprecision(c, dx1, dx2, dy1, dy2)]
        print("____________   Raw data        LD with S         LD with S&HL         Ld with HL")
        print("Precision        ", end="")
        for item in comparison_result:
            print("%4.2f%%%12s" % (item*100, ""), end="")
        print("")

    def run(self):
        self.compare_clf_and_prep(self.raw_X_train, self.raw_X_test, self.raw_Y_train, self.raw_Y_test,
                                  self.low_dim_X_train, self.low_dim_X_test, self.low_dim_Y_train,
                                  self.low_dim_Y_test)
        print("Comparison between raw data and preprocessed data in SVC with Non-linear transformation")
        self.comparison(self.raw_X_train, self.raw_X_test, self.raw_Y_train, self.raw_Y_test,
                        self.low_dim_X_train, self.low_dim_X_test, self.low_dim_Y_train,
                        self.low_dim_Y_test, self.discretized_X_train, self.discretized_X_test, self.discretized_Y_train,
                        self.discretized_Y_test)


class processor:
    def __init__(self, obj, str):
        self.obj = obj
        self.descr = str


if __name__ == '__main__':
    import time
    start_time = time.time()
    a = experiment()
    a.run()
    print("--- %d seconds ---" % (time.time() - start_time))


