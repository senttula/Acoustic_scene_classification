import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Optimize_classifier_weigths:
    def __init__(self):
        self.theta = np.array([1])
        self.alpha = 0.3
        self.xtest = np.array([1])
        self.ytest = np.array([1])
        self.xshape= np.array([1])

        self.OHencoder = OneHotEncoder()

    def train_theta(self, x_train, y_train, xtest=None, ytest=None):

        #print(x_train.shape)
        #print(y_train.shape)

        self.xtest = xtest
        self.ytest = ytest

        self.theta = np.random.rand(x_train.shape[1]*x_train.shape[2]).reshape((-1, 1))
        self.xshape=x_train.shape

        self.gradientDescent_train(reshape_x(x_train), y_train)
        #predicts = np.argmax(np.dot(self.theta, x_train), axis=1)

        return self.theta

    def gradientDescent_train(self, x, y):
        numIterations = 10000
        number_of_items = x.shape[0]
        y_transformed = self.OHencoder.fit_transform(y.reshape(-1, 1)).toarray()
        previous_cost = 0
        theta = np.random.randn(self.xshape[0])
        print ("gradient descent iterations: "), numIterations
        for i in range(0, numIterations+1):
            hypothesis = np.dot(theta, x)
            loss = hypothesis - y_transformed

            sum = np.zeros((theta.shape[0]))
            for t in range(x.shape[0]):
                loss_for_item = np.dot(x[t], loss[t])
                sum = np.add(sum, loss_for_item)
            gradient = sum / number_of_items

            theta = theta - self.alpha * gradient

            theta = np.clip(theta, 0, None)#negative weigths is overlearning

            score  =np.argmax(hypothesis, axis=1)
            acc = np.mean(score == y)
            cost = np.sum(np.power(loss, 2)) / (2 * number_of_items)
            if not i%int(numIterations/10) or np.log10(i).is_integer():
                self.theta = theta / max(abs(theta))
                print("Iteration %4d | Cost: %.8f, accuracy: %.3f"% (i, cost, acc), end="")
                self.test()
                print()
                if round(cost, 15) == round(previous_cost, 15):#break if cost did not decrease in a while
                    print("Cost stagnant, ending iterations")
                    break
                previous_cost = cost





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