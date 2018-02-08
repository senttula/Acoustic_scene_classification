import preprocessing
import simplemodels
import numpy as np
from help_functions import Optimize_classifier_weigths

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier


class main_model:
    def __init__(self, preprocess_class):
        self.simplemodels = simplemodels.simplemodels(1, preprocess_class)
        self.preprocess_class = preprocess_class

        self.classifiers = [
            (LinearDiscriminantAnalysis(), "LinearDiscriminantAnalysis"),
            (svm.SVC(kernel="linear"), "svm linear"),
            (svm.SVC(kernel="rbf"), "svm rbf"),
            (KNeighborsClassifier(), "5 NN"),
            (LogisticRegression(), "LogisticRegression"),
            (RandomForestClassifier(), "RandomForestClassifier"),
            (ExtraTreesClassifier(), "ExtraTreesClassifier"),
            (AdaBoostClassifier(), "AdaBoostClassifier"),
            (GradientBoostingClassifier(), "GradientBoostingClassifier"),
        ]

    def get_submissions(self):
        self.simplemodels.mode = 3
        submission_predicts, training_predicts = self.simplemodels.all_simple_models()

        submission_predicts, training_predicts = reshape_x(submission_predicts), reshape_x(training_predicts)

        y,_ = self.preprocess_class.get_labels()

        optimizer = Optimize_classifier_weigths()

        classfier_weigths = np.array(
            [0.15571491,  1. ,         0.74919279 , 0.07095752 , 0.17327942, - 0.29226607,
             0.23310983  , 0.28003909])

        classfier_weigths = optimizer.train_and_predict(training_predicts, y)

        print (classfier_weigths)
        print(classfier_weigths.shape)

        weigthed_submission = np.argmax(np.dot(classfier_weigths, submission_predicts), axis=1)

        print("submission shape: ", weigthed_submission.shape)
        return weigthed_submission




def reshape_x(x):
    reshaped = []
    for i in range(x.shape[1]):
        part = x[:, i, :]
        reshaped.append(part)
    return np.array(reshaped)




