import preprocessing
import simplemodels
import numpy as np
from help_functions import Optimize_classifier_weigths, reshape_x, test_by_class
import neuro_networks


class main_model:
    def __init__(self, preprocess_class):
        self.preprocess_class = preprocess_class

        self.simplemodels = simplemodels.simplemodels(1, preprocess_class)
        self.networks = neuro_networks.neuronetwork_models(preprocess_class)

        self.classfier_weigths = None

        self.best_configuration_list = [] #mask, threshold, accuracy_change


    def test_neuronets(self):
        self.networks.all_neuronetwork_models()

    def test_full(self):
        thresholds = [0.5]
        for z in range(0,self.preprocess_class.nfold):
            print("###########################################################", z)
            print("training models to test")
            self.preprocess_class.change_crossvalidation(z)
            self.simplemodels.semisupervised = False
            self.simplemodels.mode = 1
            simple_model_predicts = self.simplemodels.all_simple_models()

            simple_model_predicts = reshape_x(simple_model_predicts)

            _,y_test = self.preprocess_class.get_labels()

            if self.classfier_weigths is None:
                self.classfier_weigths  =np.ones(simple_model_predicts.shape[1])

            print("weigths: ", np.round(self.classfier_weigths, 3))

            weigths_to_test_semisupervised = np.copy(self.classfier_weigths)

            weigthed_predicts = np.argmax(np.dot(self.classfier_weigths, simple_model_predicts), axis=1)

            acc = np.mean(weigthed_predicts == y_test)
            print("test accuracy: ", acc)

            for index in range(len(thresholds)):
                threshold = thresholds[index]
                self.classfier_weigths = weigths_to_test_semisupervised
                self.preprocess_class.threshold = threshold
                weigthed_predicts = self.semisupervised(simple_model_predicts)
                acc2 = np.mean(weigthed_predicts == y_test)
                accuracy_change=(acc2-acc)/acc
                print ("accuracy_change: ", accuracy_change)
                self.best_configuration_list.append([round(accuracy_change, 4), threshold,
                                                     np.round(self.classfier_weigths, 3)])

        for t in self.best_configuration_list:
            print(t)


    def get_submissions(self):

        self.preprocess_class.threshold = 0.6
        print("###########################################################")
        print("training models to make submission")
        self.simplemodels.mode = 3
        simple_model_predicts = self.simplemodels.all_simple_models()

        simple_model_predicts = reshape_x(simple_model_predicts)

        if self.classfier_weigths is None:
            self.classfier_weigths  = np.ones(simple_model_predicts.shape[1])

        print("nonzero weigths: ", np.round(self.classfier_weigths, 3))

        #weigthed_submission = np.argmax(np.dot(self.classfier_weigths, simple_model_predicts), axis=1)

        weigthed_submission = self.semisupervised(simple_model_predicts)

        #print("submission shape: ", weigthed_submission.shape)
        return weigthed_submission

    def semisupervised(self, predicts):
        # TODO try with different thresholds
        print("###########################################################", self.preprocess_class.threshold)
        print("training again with semisupervised data")
        weigthed_predicts_probas = np.dot(self.classfier_weigths, predicts)
        weigthed_predicts_probas = weigthed_predicts_probas / np.sum(self.classfier_weigths)  # normalize
        self.preprocess_class.predict_probabilities = weigthed_predicts_probas

        self.simplemodels.semisupervised = True

        self.simplemodels.reset_mask()
        self.train_classifier_weigths()

        new_simple_model_predicts = self.simplemodels.all_simple_models()
        new_simple_model_predicts = reshape_x(new_simple_model_predicts)
        weigthed_predicts = np.argmax(np.dot(self.classfier_weigths, new_simple_model_predicts), axis=1)

        return weigthed_predicts

    def train_classifier_weigths(self):
        #self.classfier_weigths = [1.,  1. ,  1.,  1.]
        #return
        for z in range(0,self.preprocess_class.nfold):
            print("###########################################################")
            print("training models to optimize classifier weigths")
            self.preprocess_class.change_crossvalidation(z)
            self.simplemodels.semisupervised = False
            self.simplemodels.mode = 1
            train_predicts = self.simplemodels.all_simple_models()
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




"""
[[0.021299999999999999, 0.61, array([ 1.,  1.,  0.,  0.,  1.,  0.])],
[0.0436, 0.61, array([-0.002,  0.215,  0.208,  0.334,  1.   ,  0.236])],
 [0.050500000000000003, 0.61, array([ 1.,  1.,  0.,  1.,  0.,  0.])],
  [0.066500000000000004, 0.61, array([ 1.,  1.,  1.,  0.,  0.,  1.])]]




"""




