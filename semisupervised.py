from sklearn.metrics import accuracy_score
import numpy as np

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

from help_functions import test_by_class
from time import  time






class semisupervised:
    def __init__(self):
        self.threshold = 0.5



    def learn(self, x_all, y_all, x_submission, predict_probabilities):
        """
        detect high probability labels from predict_probabilities
        add these to train data with the likely label
        train, predict again, win
        """

        y_all = y_all.reshape(-1, 1)

        probas = predict_probabilities

        new_set = np.where(probas > self.threshold) #tuple: x indexes, y indexes
        new_x = x_submission[new_set[0]]
        new_y = new_set[1].reshape(-1, 1)

        new_x_all = np.concatenate((new_x, x_all))
        new_y_all = np.concatenate((new_y, y_all))

