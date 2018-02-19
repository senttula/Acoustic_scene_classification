import numpy as np
from sklearn.preprocessing import OneHotEncoder
from time import time
import matplotlib.pyplot as plt

class Optimize_classifier_weigths:
    def __init__(self):
        self.theta = np.array([1])
        self.alpha = 0.1
        self.xtest = np.array([1])
        self.ytest = np.array([1])

        self.max_iterations = 2000

        self.c = 0 # TODO regularisation

        self.OHencoder = OneHotEncoder()

    def train_theta(self, x_train, y_train, xtest=None, ytest=None):

        #print(x_train.shape)
        #print(y_train.shape)

        self.xtest = xtest
        self.ytest = ytest

        theta_candidates = []

        print("---------gradient descent iterations with different loss functions, iterations max: ",
              self.max_iterations, ", learn rate: ",self.alpha,"----------")

        theta_candidates.append(forward_recursive_selection(x_train, y_train))
        theta_candidates.append(backward_recursive_selection(x_train, y_train))
        #print(theta_candidates)
        #theta_candidates.append(self.gradientDescent_train(x_train, y_train, self.loss_full))
        # theta_candidates.append(binary_weigths(x_train, y_train))

        accuracies = []
        for theta_test in theta_candidates:
            # test all theta candidates, to choose
            hypothesis = np.dot(theta_test, x_train)
            score = np.argmax(hypothesis, axis=1)
            acc = np.mean(score == y_train)
            accuracies.append(acc)


        best_theta = theta_candidates[np.argmax(np.array(accuracies))]#select best by accuracy
        print ("best accuracy: %.4f , weigths: " %(max(accuracies)), best_theta)
        return best_theta

    def gradientDescent_train(self, x, y, loss_function):
        number_of_items = x.shape[0]
        y_transformed = self.OHencoder.fit_transform(y.reshape(-1, 1)).toarray()
        previous_cost = 0
        theta = np.ones(x.shape[1])

        for i in range(1, self.max_iterations+1):
            hypothesis = np.dot(theta, x)

            loss = loss_function(hypothesis, y_transformed, y)

            sum = np.zeros((theta.shape[0]))
            for t in range(x.shape[0]): #TODO find if any numpy way to do this
                loss_for_item = np.dot(x[t], loss[t])
                sum = np.add(sum, loss_for_item)

            gradient = sum / number_of_items
            gradient_with_penalty = gradient - self.c * theta
            theta = theta - self.alpha * gradient_with_penalty

            gradient = sum / number_of_items

            theta = theta - self.alpha * gradient

            theta = theta / max(abs(theta))

            #theta = np.clip(theta, 0, None)#negative weigths is overlearning?

            cost = np.sum(np.power(loss, 2)) / (2 * number_of_items)

            if not i%int(self.max_iterations/10) or i == 10 or i == 100:

                score = np.argmax(hypothesis, axis=1)
                acc = np.mean(score == y)

                cost_delta = previous_cost - cost
                #cost_delta = np.log10(cost_delta)
                print("Iteration %5d | Cost decrease: %.3e, accuracy: %.4f " % (i, cost_delta, acc), end="")
                self.test()
                print()

                # print(theta)
                # print(gradient / max(abs(gradient)))
                # print ((self.c * theta) / max(abs(self.c * theta)))

                if abs(cost_delta) < 1e-16:#break if cost did not decrease in a while
                    print("Cost stagnant, ending iterations")
                    break

            previous_cost = cost

        theta = theta / max(abs(theta))

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

def binary_weigths(x, y):
    # replaced by recursive selection, too slow
    #TODO are all important? if classifiers>15 gets real slow
    print("testing with 0/1 weigths (binary).. ")
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
            # print (n, theta)
            best=(theta.copy(), acc) #theta needs to be copied
    print("best binary weigthed accuracy: ", round(best[1],4))
    return best[0]

def forward_recursive_selection(x, y):
    # appends the best until lower accuracy
    best_indexes = []
    best_accuracy = 0
    indexes_to_check = list(range(x.shape[1]))
    for n in range(x.shape[1]):
        accuracies = []
        for i in indexes_to_check:
            theta = np.zeros(x.shape[1])
            theta[best_indexes] = 1
            theta[i] = 1
            hypothesis = np.dot(theta, x)
            score = np.argmax(hypothesis, axis=1)
            acc = np.mean(score == y)
            accuracies.append(acc)

        if max(accuracies) < best_accuracy or n==x.shape[1]-1:
            theta = np.zeros(x.shape[1])
            theta[best_indexes] = 1
            return theta
        best = indexes_to_check[np.argmax(accuracies)]
        best_accuracy = max(accuracies)
        best_indexes.append(best)
        indexes_to_check.remove(best)

    theta = np.zeros(x.shape[1])
    theta[best_indexes] = 1
    return theta

def backward_recursive_selection(x, y):
    # appends the best until lower accuracy
    best_indexes = []
    best_accuracy = 0
    indexes_to_check = list(range(x.shape[1]))
    for n in range(x.shape[1]):
        accuracies = []
        for i in indexes_to_check:
            theta = np.ones(x.shape[1])
            theta[best_indexes] = 0
            theta[i] = 0
            hypothesis = np.dot(theta, x)
            score = np.argmax(hypothesis, axis=1)
            acc = np.mean(score == y)
            accuracies.append(acc)
        if max(accuracies) < best_accuracy or n==x.shape[1]-1:
            theta = np.ones(x.shape[1])
            theta[best_indexes] = 0
            return theta
        best = indexes_to_check[np.argmax(accuracies)]
        best_accuracy = max(accuracies)
        best_indexes.append(best)
        indexes_to_check.remove(best)

    theta = np.ones(x.shape[1])
    theta[best_indexes] = 0
    return theta




def test_by_class(predicts, y):
    print ("testing predicts by class...")
    number_of_classes = 15
    info = {i:[0,0,0] for i in range (number_of_classes)}
    for index in range(predicts.shape[0]):
        predict = predicts[index]
        correct = y[index]
        if predict == correct:
            info[predict][0] = info[predict][0]+1
        else:
            info[predict][1] = info[predict][1]+1
            info[correct][2] = info[correct][2]+1

    for index in range(number_of_classes):
        correct_predictions = info[index][0]
        false_positives = info[index][1]
        missed_positives = info[index][2]

        total_occurances  = correct_predictions+missed_positives
        total_predictions = correct_predictions+false_positives
        try:
            false_chance = false_positives / total_predictions
            missing_correct_chance = missed_positives / total_occurances


            #print (index,": ",info[index][0], info[index][1], info[index][2], end="    ")
            #print("false chance: ",round(false_positives / total_occurances, 3), end=" ")
            #print("missing correct chance: ", round(missed_positives / total_predictions, 3), end=" ")


            #print("class index: %2d correct predictions: %2d false positives: %2d missed positives: %2d \n"
            #      "\tfalse chance: %.3f ""missing correct chance: %.3f" %(index,correct_predictions,false_positives,
            #                missed_positives, false_chance, missing_correct_chance))

            #print("cls%.2d %.3f " % (index,correct_predictions/(total_predictions)))
            print("cls%.2d %.3f %.3f %.3f" % (index, false_chance, missing_correct_chance,
                                              correct_predictions / (total_occurances + false_positives)))

            #print(round((info[index][0] + info[index][2]) / (info[index][0]+info[index][1] + info[index][2]), 3), end=" ")
            #print(round(info[index][0] / (info[index][0] + info[index][2]), 3), end=" ")
        except:pass






