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


    def test_neuronets(self):
        self.networks.all_neuronetwork_models()

    def test_full(self):
        for z in range(5):
            self.preprocess_class.change_crossvalidation(z)
            print("###########################################################")
            print("training models to test")
            self.simplemodels.mode = 1
            simple_model_predicts = self.simplemodels.all_simple_models()

            simple_model_predicts = reshape_x(simple_model_predicts)

            _,y_test = self.preprocess_class.get_labels()

            if self.classfier_weigths is None:
                self.classfier_weigths  =np.ones(simple_model_predicts.shape[1])

            print("weigths: ", np.round(self.classfier_weigths, 3))

            weigthed_predicts = np.argmax(np.dot(self.classfier_weigths, simple_model_predicts), axis=1)

            acc = np.mean(weigthed_predicts == y_test)
            print("test accuracy: ", acc)

            weigthed_predicts = self.semisupervised(simple_model_predicts)
            acc2 = np.mean(weigthed_predicts == y_test)

            print("test accuracy: ", acc)
            print("test accuracy: ", acc2, " after semisupervised learning")
            # TODO plot accuracy graph



    def get_submissions(self):
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

    def semisupervised(self, simple_model_predicts):
        # TODO try with different thresholds
        print("###########################################################")
        print("training again with semisupervised data")
        weigthed_predicts_probas = np.dot(self.classfier_weigths, simple_model_predicts)
        weigthed_predicts_probas = weigthed_predicts_probas / np.sum(self.classfier_weigths)  # normalize
        self.preprocess_class.predict_probabilities = weigthed_predicts_probas

        self.simplemodels.semisupervised = True

        #self.train_classifier_weigths() # TODO

        new_simple_model_predicts = self.simplemodels.all_simple_models()
        new_simple_model_predicts = reshape_x(new_simple_model_predicts)
        weigthed_predicts = np.argmax(np.dot(self.classfier_weigths, new_simple_model_predicts), axis=1)

        return weigthed_predicts

    def train_classifier_weigths(self):
        print ("###########################################################")
        print ("training models to optimize classifier weigths")
        self.simplemodels.mode = 1
        train_predicts = self.simplemodels.all_simple_models()
        _, y = self.preprocess_class.get_labels()

        #TODO add test data into predicts

        #TODO how weigths would come if trained on train+test data
        optimizer = Optimize_classifier_weigths()
        self.classfier_weigths = optimizer.train_theta(train_predicts, y)#, x_test, y_test)

        print("weigths: ",np.round(self.classfier_weigths, 3))

        obsolete_classifiers_indexes  =np.where(self.classfier_weigths == 0)[0]
        index_obsolete=0
        for index in range(len(self.simplemodels.classifiers_mask)):
            if self.simplemodels.classifiers_mask[index]:
                if index_obsolete in obsolete_classifiers_indexes:
                    self.simplemodels.classifiers_mask[index] = 0
                index_obsolete +=1

        self.classfier_weigths = np.delete(self.classfier_weigths, obsolete_classifiers_indexes)

        print()
        print()










