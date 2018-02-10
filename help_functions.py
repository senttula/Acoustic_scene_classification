import numpy as np
from sklearn.preprocessing import OneHotEncoder
from time import time

class Optimize_classifier_weigths:
    def __init__(self):
        self.theta = np.array([1])
        self.alpha = 0.2 #TODO needs to be lower if more classifiers
        self.xtest = np.array([1])
        self.ytest = np.array([1])
        self.xshape= np.array([1])

        self.max_iterations = 5000

        self.OHencoder = OneHotEncoder()

    def train_theta(self, x_train, y_train, xtest=None, ytest=None):

        #print(x_train.shape)
        #print(y_train.shape)

        self.xtest = xtest
        self.ytest = ytest

        self.xshape=x_train.shape
        theta_candidates = []
        reshaped_x = reshape_x(x_train)
        print("---------gradient descent iterations with different loss functions, ",
              self.max_iterations, "iterations max----------")

        theta_candidates.append(self.gradientDescent_train(reshaped_x, y_train, self.loss_full))

        # these loss functions are worse and slow
        #theta_candidates.append(self.gradientDescent_train(reshaped_x, y_train, self.loss_only_best))
        #theta_candidates.append(self.gradientDescent_train(reshaped_x, y_train, self.loss_two_best))

        theta_candidates.append(binary_weigths(reshaped_x, y_train))

        accuracies = []
        for theta_test in theta_candidates:
            #test all theta candidates again as backup
            hypothesis = np.dot(theta_test, reshaped_x)
            score = np.argmax(hypothesis, axis=1)
            acc = np.mean(score == y_train)
            accuracies.append(acc)

        best_theta = theta_candidates[np.argmax(np.array(accuracies))]#select best by accuracy
        return best_theta

    def gradientDescent_train(self, x, y, loss_function):
        number_of_items = x.shape[0]
        y_transformed = self.OHencoder.fit_transform(y.reshape(-1, 1)).toarray()
        previous_cost = 0
        theta = np.random.randn(self.xshape[0])
        for i in range(1, self.max_iterations+1):
            hypothesis = np.dot(theta, x)

            loss = loss_function(hypothesis, y_transformed, y)

            sum = np.zeros((theta.shape[0]))
            for t in range(x.shape[0]): #TODO find if any numpy way to do this
                loss_for_item = np.dot(x[t], loss[t])
                sum = np.add(sum, loss_for_item)
            gradient = sum / number_of_items

            theta = theta - self.alpha * gradient

            theta = np.clip(theta, 0, None)#negative weigths is overlearning

            score  =np.argmax(hypothesis, axis=1)
            acc = np.mean(score == y)
            cost = np.sum(np.power(loss, 2)) / (2 * number_of_items)
            if not i%int(self.max_iterations/10) or i == 10 or i == 100:
                theta = theta / max(abs(theta))
                print("Iteration %4d | Cost: %.8f, accuracy: %.3f"% (i, cost, acc), end="")
                self.test()
                print()
                if round(cost, 15) == round(previous_cost, 15):#break if cost did not decrease in a while
                    print("Cost stagnant, ending iterations")
                    break
                previous_cost = cost
        return theta

    def loss_full(self, hypothesis, y_transformed, y):
        return hypothesis - y_transformed

    def loss_only_best(self, hypothesis, y_transformed, y):
        index_max = np.argmax(hypothesis, axis=1)
        for itemindex in range(hypothesis.shape[0]):
            for classf in range(hypothesis.shape[1]):
                if classf != index_max[itemindex] and classf != y[itemindex]:
                    hypothesis[itemindex][classf] = 0
        return hypothesis - y_transformed

    def loss_two_best(self, hypothesis, y_transformed, y):
        index_max = np.argmax(hypothesis, axis=1)

        mask = np.zeros_like(hypothesis)
        mask[index_max] = 1
        masked_hypothesis = np.ma.masked_array(hypothesis, mask=mask)
        index_max_second = np.argmax(masked_hypothesis, axis=1)

        for itemindex in range(hypothesis.shape[0]):
            for classf in range(hypothesis.shape[1]):
                if classf != index_max[itemindex] and classf != y[itemindex] and classf != index_max_second[itemindex]:
                    hypothesis[itemindex][classf] = 0
        return hypothesis - y_transformed

    def test(self):
        if self.xtest is not None and self.ytest is not None :
            predicts = np.argmax(np.dot(self.theta, self.xtest), axis=1)
            acc = np.mean(predicts == self.ytest)
            print(", test accurcy: %.3f" % (acc), end="")


def reshape_x(x):
    reshaped = []
    for i in range(x.shape[1]):
        part = x[:, i, :]
        reshaped.append(part)
    return np.array(reshaped)

def binary_weigths(x, y):
    theta = np.zeros(x.shape[1])
    best = (theta, 0.0)  # (theta values, accuracy)
    n = 2**x.shape[1]
    for theta_int in range(1,n):
        theta_str = str(bin(theta_int))
        for i in range(2, len(theta_str)):
            theta[i-2]=int(theta_str[i])
        hypothesis = np.dot(theta, x)
        score = np.argmax(hypothesis, axis=1)
        acc = np.mean(score == y)
        if acc>best[1]:
            best=(theta.copy(), acc) #theta needs to be copied
    print("best binary: accuracy: ", round(best[1],4))
    return best[0]

