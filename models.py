import preprocessing
import simplemodels
import numpy as np
from sklearn.metrics import accuracy_score
from random import random, uniform
import test_model
#import tflearn
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.layers.estimator import regression
#from tflearn.data_utils import shuffle, to_categorical

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

from help_functions import Optimize_classifier_weigths

class main_model:
    def __init__(self, preprocess_class):
        self.simplemodels = simplemodels.simplemodels(1, preprocess_class)
        self.preprocess = preprocess_class

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

    def get_models(self):
        n_classifiers = len(self.simplemodels.classifiers)


    def get_submissions(self):
        self.simplemodels.mode = 3
        x1 = self.simplemodels.all_simple_models()
        reshaped_x = reshape_x(x1)

        classfier_weigths = np.array(
            [0.15571491,  1. ,         0.74919279 , 0.07095752 , 0.17327942, - 0.29226607,
             0.23310983 , 0.04205986 , 0.28003909]
                                     )

        submission = np.argmax(np.dot(classfier_weigths, reshaped_x), axis=1)

        print("submission shape: ", submission.shape)
        return submission



    def testausta(self):
        self.simplemodels.mode = 3
        #x1 = self.simplemodels.all_simple_models()
        #reshaped_x1 = reshape_x(x1)
        #y2, y1 = self.preprocess.get_labels()
#
        #self.simplemodels.mode = 2
        #x2 = self.simplemodels.all_simple_models()
        #reshaped_x2 = reshape_x(x2)

        self.preprocess.is_submission = True

        x1, x2 = self.preprocess.mean_time()
        y, _ = self.preprocess.get_labels()

        print(x1.shape)
        print(x2.shape)
        print(y.shape)

        clf = svm.SVC(kernel="rbf", probability=True, C=15, gamma=.029, tol=0.1)



        clf.fit(x1, y)


        return clf.predict(x2)


        reshaped_x1 = np.load("rstest.npy")
        reshaped_x2 = np.load("rstest2.npy")
        y1 = np.load("rstesty1.npy")
        y2 = np.load("rstesty2.npy")



        tt = Optimize_classifier_weigths()
        cw = tt.train_and_predict(reshaped_x1, y1,reshaped_x2, y2)

        print (cw)

        #classfier_weigths = np.array(
        #    [0.53714105,  0.1601243 ,  1., - 0.09961459 , 0.15059309,  0.1359336,
        #     - 0.45813827,  0.38088524 , 0.57047298]
        #                             )
#
        #print("x shape: ",reshaped_x.shape)
        ##submission = np.argmax(np.dot(classfier_weigths, reshaped_x), axis=1)
        #gradient_descent = gradient_descent_class()
#
        #submission  =gradient_descent.train_and_predict(reshaped_x, y1, reshaped_x2, y2)
#
        #print("submission shape: ", submission.shape)
        #return submission
#
#
        #n_classifiers = len(self.simplemodels.classifiers)
#
        #self.simplemodels.mode = 1
        ##x1 = self.simplemodels.all_simple_models()
        #y2, y1 = self.preprocess.get_labels()
#
        ##self.simplemodels.mode = 2
        ##x2 = self.simplemodels.all_simple_models()
#
        #reshaped1 = np.load("rstest.npy")  # reshape_x(x1)
#
        ##classfier_weigths = np.array(
        #    [0.53714105,  0.1601243 ,  1., - 0.09961459 , 0.15059309,  0.1359336,
        #     - 0.45813827,  0.38088524 , 0.57047298]
        #                             )
#
        #classfier_weigths = np.array(
        #    [0.78368628 , 0.57143857 , 1.   ,       0.00251855 , 0.38716152 , 0.03240557,
        #     0.01597293,  0.62785178,  0.61911972]
        #                             )
        classfier_weigths = np.array(
        [0.76253996 , 0.55642483,  1.     ,     0.       ,   0.36830787 , 0.    ,      0.,
         0.62721707 , 0.62295443]
        )


        #print(x1.shape)
        #print(x2.shape)
        #print(y1.shape)
        #print(y2.shape)

        reshaped1 = np.load("rstest.npy")#reshape_x(x1)
        print(classfier_weigths.shape)




        quit()
        #print(reshaped1.shape)
        #predicts = np.dot(classfier_weigths, reshaped1[0])#np.argmax(, axis=1)
        #print (predicts.shape)
        #mx = max(predicts)
        #for prd in predicts:
        #    print(round(prd/mx, 2))
#
        #brdnm =classfier_weigths
        #predicts = np.argmax(np.dot(brdnm, reshaped1), axis=1)
        #bacc = 0# accuracy_score(y1, predicts)


        #kaik  =[]
        #for i in range (10000):
        #    rdnm = np.random.normal(loc=brdnm, scale=0.01, size=None)
#
        #    rdnm /= np.max(rdnm)
        #    rdnm = np.where(rdnm > 0.04, rdnm, 0)
#
        #    #rdnm = np.clip(rdnm, 0, 1)
        #    predicts = np.argmax(np.dot(rdnm, reshaped1), axis=1)
        #    acc = accuracy_score(y1, predicts)
        #    if acc > bacc:
        #        print(round(acc, 5), round(bacc, 5))
        #        brdnm=rdnm
        #        bacc = acc
        #    elif acc == bacc:
        #        kaik.append(rdnm)
#
        #print ("#")
        #print (len(kaik))
        #print(np.mean(np.array(kaik), axis=0))
        #print(np.std(np.array(kaik), axis=0))
        #print("#")
#
#
        #predicts = np.argmax(np.dot(brdnm, reshaped1), axis=1)
        #acc = accuracy_score(y1, predicts)
        #print (acc)
        #print (brdnm)
        #print ("##")
        #quit()
#
        #puol = np.dot(np.linalg.inv(np.dot(reshaped1.T, reshaped1)), reshaped1.T)
        #theta = np.dot(puol, y1)
        #print (theta)
        #quit()

        #np.save("rstest.npy", reshaped1)

        #lg = [0, 1]
        #acg = []
        #wgt = []
        #for a in lg:
        #    for b in lg:
        #        print (b)
        #        for c in lg:
        #            for d in lg:
        #                for e in lg:
        #                    for f in lg:
        #                        for g in lg:
        #                            for h in lg:
        #                                for i in lg:
        #                                    cw = np.array([i, h, g, f, e, d, c, b, a])
        #                                    predicts = np.argmax(np.dot(cw, reshaped1), axis=1)
        #                                    acg.append(accuracy_score(y1, predicts))
        #                                    wgt.append(cw)
        #mx = max(acg)
        #print (mx)
        #for i in range(len(acg)):
        #    ac = acg[i]
        #    #if ac == mx:
        #    print(round (ac,3) , end=" ")
        #    print (wgt[i])





        #predicts = np.argmax(np.dot(classfier_weigths, reshaped1), axis=1)
        #accuracy = accuracy_score(y1, predicts)
        #print(accuracy)

        #reshaped2 = reshape_x(x2)
        #predicts  = np.argmax(np.dot(classfier_weigths, reshaped2), axis=1)
        #accuracy = accuracy_score(y2, predicts)
        #print (accuracy)


        submission = predicts
        return submission



def reshape_x(x):
    reshaped = []
    for i in range(x.shape[1]):
        part = x[:, i, :]
        reshaped.append(part)
    return np.array(reshaped)







def neuro():
    network = input_data(shape=[None, n_classifiers * 15])
    network = fully_connected(network, 15, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    self.model = tflearn.DNN(network, tensorboard_verbose=0)

    # print (x1.shape)
    # print(x2.shape)
    # print(y1.shape)
    # print(y2.shape)

    ohe = OneHotEncoder(sparse=False)
    y1 = ohe.fit_transform(y1.reshape(-1, 1))
    y2 = ohe.transform(y2.reshape(-1, 1))

    print(x1.shape)
    print(x2.shape)
    print(y1.shape)
    print(y2.shape)
    # print (x1)

    # x1 = x1[:, 1]


    # x1 = np.expand_dims(x1, axis=2)
    x1 = x1.reshape(-1, 1)
    x1 = ohe.transform(x1)
    x1 = x1.reshape(-1, n_classifiers * 15)

    x2 = x2.reshape(-1, 1)
    x2 = ohe.transform(x2)
    x2 = x2.reshape(-1, n_classifiers * 15)

    # print(x1)

    # x1  = np.expand_dims(x1, axis=2)
    # x2 = np.expand_dims(x2, axis=2)
    # print(x1)
    # print (y1)

    self.model.fit(x1,
                   y1,
                   n_epoch=10,
                   batch_size=32,
                   show_metric=True,
                   validation_set=(x2, y2)
                   )

    self.model.save("modeltest67o.tfl")



    # return submission





