import preprocessing
import simplemodels
import numpy as np
from help_functions import Optimize_classifier_weigths, test_by_class
#import neuro_networks


class main_model:
    def __init__(self, preprocess_class):
        self.preprocess_class = preprocess_class

        self.simplemodels = simplemodels.simplemodels(preprocess_class)
        #self.networks = neuro_networks.neuronetwork_models(preprocess_class)

        self.classfier_weigths = None

        self.best_configuration_list = [] #mask, threshold, accuracy_change


    #def test_neuronets(self):
    #    self.networks.all_neuronetwork_models()

    def get_predict_probas(self, is_submission = False, crossvalidation_fold = -1):
        if is_submission:
            self.simplemodels.is_submission = True
        elif crossvalidation_fold != -1: #if submission, crossvalidation doesn't matter
            self.preprocess_class.change_crossvalidation(crossvalidation_fold)

        all_simple_probas = self.simplemodels.all_simple_models()
        # TODO same for neuromodels
        # probas = np.concentate((s, n), axis = 1)

        probas = all_simple_probas
        return probas

    def test_full(self):
        thresholds = [0.5, 0.55, 0.6, 0.65]
        for z in range(0,1):#self.preprocess_class.nfold):
            print("###########################################################", z)
            print("training models to test")

            simple_model_predicts = self.get_predict_probas(crossvalidation_fold=z)

            _,y_test = self.preprocess_class.get_labels()

            if self.classfier_weigths is None:
                self.classfier_weigths  =np.ones(simple_model_predicts.shape[1])

            print("weigths: ", np.round(self.classfier_weigths, 3))

            weigths_to_test_semisupervised = np.copy(self.classfier_weigths)

            weigthed_predicts = np.argmax(np.dot(self.classfier_weigths, simple_model_predicts), axis=1)

            acc = np.mean(weigthed_predicts == y_test)
            print("test accuracy: ", acc)

            acl = []
            acl.append(acc)
            for index in range(len(thresholds)):
                acl.append(0)
                self.classfier_weigths = weigths_to_test_semisupervised
                self.preprocess_class.threshold = thresholds[index]

                probas = self.semisupervised(simple_model_predicts)
                weigthed_predicts = np.argmax(np.dot(self.classfier_weigths, probas), axis=1)
                acc2 = np.mean(weigthed_predicts == y_test)
                accuracy_change=(acc2-acc)/acc
                print ("accuracy_change: ", accuracy_change,acc2)
                acl.append(acc2)

                probas = self.semisupervised(probas)
                weigthed_predicts = np.argmax(np.dot(self.classfier_weigths, probas), axis=1)
                acc3 = np.mean(weigthed_predicts == y_test)
                accuracy_change = (acc3 - acc2) / acc2
                print("accuracy_change: ", accuracy_change, acc3)
                acl.append(acc3)
#
                #probas = self.semisupervised(probas)
                #weigthed_predicts = np.argmax(np.dot(self.classfier_weigths, probas), axis=1)
                #acc4 = np.mean(weigthed_predicts == y_test)
                #accuracy_change = (acc4 - acc3) / acc3
                #print("accuracy_change: ", accuracy_change, acc4)
                #acl.append(acc4)


        for t in acl:
            print(t)

    def get_submissions(self):
        self.preprocess_class.threshold = 0.6
        print("###########################################################")
        print("training models to make submission")

        simple_model_predicts = self.get_predict_probas(is_submission=True)
        if self.classfier_weigths is None:
            self.classfier_weigths  = np.ones(simple_model_predicts.shape[1])

        print("nonzero weigths: ", np.round(self.classfier_weigths, 3))

        print (simple_model_predicts.shape)
        weigthed_submission = self.semisupervised(simple_model_predicts)
        print(weigthed_submission.shape)
        weigthed_submission = self.semisupervised(weigthed_submission)

        weigthed_submission = np.argmax(np.dot(self.classfier_weigths, weigthed_submission), axis=1)
        #print("submission shape: ", weigthed_submission.shape)
        return weigthed_submission

    def semisupervised(self, probas):
        # TODO try with different thresholds
        print("###########################################################", self.preprocess_class.threshold)
        print("training again with semisupervised data")
        weigthed_predicts = np.dot(self.classfier_weigths, probas)
        weigthed_predicts = weigthed_predicts / np.sum(self.classfier_weigths)  # normalize, problems if some weigth <0

        self.preprocess_class.init_semisupervised(weigthed_predicts)
        new_probas = self.get_predict_probas()
        self.preprocess_class.reset_semisupervised()

        return new_probas

    def train_classifier_weigths(self):
        for z in range(0,1):#self.preprocess_class.nfold):
            print("###########################################################")
            print("training models to optimize classifier weigths")
            self.preprocess_class.change_crossvalidation(z)
            train_predicts = self.get_predict_probas()
            _, y = self.preprocess_class.get_labels()

            #TODO add test data into predicts?

            optimizer = Optimize_classifier_weigths()
            self.classfier_weigths = optimizer.train_theta(train_predicts, y)#, x_test, y_test)
            print("weigths: ",np.round(self.classfier_weigths, 3))
            print()
        #TODO show all weigths and then the choosed ones with explanation why

    def update_mask(self):
        obsolete_classifiers_indexes = np.where(self.classfier_weigths == 0)[0]
        index_obsolete = 0
        for index in range(len(self.simplemodels.classifiers_mask)):
            if self.simplemodels.classifiers_mask[index]:
                if index_obsolete in obsolete_classifiers_indexes:
                    self.simplemodels.classifiers_mask[index] = 0
                index_obsolete += 1
        self.classfier_weigths = np.delete(self.classfier_weigths, obsolete_classifiers_indexes)

