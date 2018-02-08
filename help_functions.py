from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from random import uniform
from sklearn.preprocessing import OneHotEncoder

class Optimize_classifier_weigths:
    def __init__(self):
        self.theta = np.array([1])
        self.alpha = 2
        self.xtest = np.array([1])
        self.ytest = np.array([1])
        self.OHencoder = OneHotEncoder()
        self.xshape=0

    def train_and_predict(self, x_train, y_train, xtest=None, ytest=None):

        self.xtest = xtest
        self.ytest = ytest

        self.theta = np.random.rand(x_train.shape[1]*x_train.shape[2]).reshape((-1, 1))
        self.xshape=x_train.shape

        self.gradientDescent_train(x_train, y_train)

        predicts = np.argmax(np.dot(self.theta, x_train), axis=1)

        return predicts

    def gradientDescent_train(self, x, y):
        numIterations = 4000
        alpha = 0.35
        number_of_items = x.shape[0]
        y_transformed = self.OHencoder.fit_transform(y.reshape(-1, 1)).toarray()

        theta = np.random.randn(9)
        print ("gradient descent iterations: "), numIterations
        for i in range(0, numIterations+1):
            hypothesis = np.dot(theta, x)
            loss = hypothesis - y_transformed

            sum = np.zeros((theta.shape[0]))
            for t in range(x.shape[0]):
                loss_for_item = np.dot(x[t], loss[t])
                sum = np.add(sum, loss_for_item)
            gradient = sum / number_of_items

            theta = theta - alpha * gradient

            score  =np.argmax(hypothesis, axis=1)
            acc = np.mean(score == y)
            cost = np.sum(np.power(loss, 2)) / (2 * number_of_items)
            if not i%int(numIterations/10) or np.log10(i).is_integer():
                self.theta = theta / max(abs(theta))
                print("Iteration %6d | Cost: %.6f, accuracy: %.3f"% (i, cost, acc), end="")
                self.test()
                print()

    def test(self):
        if self.xtest is not None and self.ytest is not None :
            predicts = np.argmax(np.dot(self.theta, self.xtest), axis=1)
            acc = np.mean(predicts == self.ytest)
            print(", test accurcy: %.3f" % (acc), end="")




"""
[ 0.15571382  1.          0.74919379  0.0709597   0.17328003 -0.29222287
  0.23306445  0.04206051  0.28003598]

[ 0.15571617  1.          0.74919162  0.07095499  0.17327872 -0.29231607
  0.23316235  0.0420591   0.28004269]

[ 0.15571512  1.          0.74919259  0.07095709  0.17327931 -0.29227444
  0.23311862  0.04205973  0.28003969]

"""





