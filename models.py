import preprocessing
import simplemodels
import numpy as np
from help_functions import Optimize_classifier_weigths, test_by_class
import neuro_networks


class main_model:
    def __init__(self, preprocess_class):
        self.preprocess_class = preprocess_class

        self.simplemodels = simplemodels.simplemodels(preprocess_class)
        self.networks = neuro_networks.neuronetwork_models(preprocess_class)

        self.classfier_weigths = None

        self.best_configuration_list = [] #mask, threshold, accuracy_change


    def get_predict_probas(self, is_submission = False, crossvalidation_fold = -1):
        # returns probas: shape = (??00, 9, 15)

        if is_submission:
            self.simplemodels.is_submission = True
            self.networks.is_submission = True
        else:
            self.simplemodels.is_submission = False
            self.networks.is_submission = False
            if crossvalidation_fold != -1: # crossvalidation_fold matters only if not submission
                self.preprocess_class.change_crossvalidation(crossvalidation_fold)

        all_simple_probas = self.simplemodels.all_simple_models()

        all_neuro_probas = self.networks.all_neuronets()

        probas = np.concatenate((all_simple_probas, all_neuro_probas), axis = 1)

        return probas

    def test_full(self):
        self.preprocess_class.split_crossvalidation_metatrain(10)
        for z in range(0,self.preprocess_class.nfold):
            print("###########################################################", z)
            print("training models to test")

            simple_model_predicts = self.get_predict_probas(crossvalidation_fold=z)

            _,y_test = self.preprocess_class.get_labels()

            if self.classfier_weigths is None:
                self.classfier_weigths  =np.ones(simple_model_predicts.shape[1])

            print("weigths: ", np.round(self.classfier_weigths, 3))

            weigthed_predicts = np.argmax(np.dot(self.classfier_weigths, simple_model_predicts), axis=1)
            acc = np.mean(weigthed_predicts == y_test)
            print("test accuracy: ", acc)

            #prev_acc = acc
            #probas = simple_model_predicts
            #for _ in range(2):
            #    self.preprocess_class.threshold= 0.5
            #    self.preprocess_class.confidence_threshold = 0.3
            #    probas = self.semisupervised(probas)
            #    weigthed_predicts = np.argmax(np.dot(self.classfier_weigths,probas), axis=1)
            #    new_acc = np.mean(weigthed_predicts == y_test)
            #    accuracy_change=(new_acc-prev_acc)/prev_acc
            #    print ("accuracy_change: ", accuracy_change,new_acc)
            #    accuracies.append(new_acc)
            #    prev_acc  = new_acc
            #accuracies.append(0)


    def get_submissions(self):
        print("###########################################################")
        print("training models to make submission")

        weigthed_probas= self.get_predict_probas(is_submission=True)

        if self.classfier_weigths is None:
            self.classfier_weigths = np.ones(weigthed_probas.shape[1])

        self.classfier_weigths[-1] = 3 #
        print("nonzero weigths: ", np.round(self.classfier_weigths, 3))

        for _ in range(2):# few loops of semisupervision
            weigthed_probas = self.semisupervised(weigthed_probas)

        weigthed_submission = np.argmax(np.dot(self.classfier_weigths, weigthed_probas), axis=1)

        return weigthed_submission

    def semisupervised(self, probas):
        # inputs probas, outputs probas
        print("###########################################################", self.preprocess_class.threshold)
        print("training again with semisupervised data")
        weigthed_predicts = np.dot(self.classfier_weigths, probas)
        weigthed_predicts = weigthed_predicts / np.sum(self.classfier_weigths)  # normalize, problems if some weigth <0

        self.preprocess_class.init_semisupervised(weigthed_predicts)
        new_probas = self.get_predict_probas()
        self.preprocess_class.reset_semisupervised()

        return new_probas

    def train_classifier_weigths(self):
        weigth_candidates = []
        self.preprocess_class.split_crossvalidation_metatrain(10)

        for z in range(0,self.preprocess_class.nfold):
            print("###########################################################")
            print("training models to optimize classifier weigths")
            train_predicts = self.get_predict_probas(crossvalidation_fold=z)
            _, y = self.preprocess_class.get_labels()

            #TODO add test data into predicts?

            optimizer = Optimize_classifier_weigths()
            classfier_weigths = optimizer.train_theta(train_predicts, y)#, x_test, y_test)
            print()
            weigth_candidates.append(classfier_weigths)
        for w in weigth_candidates:
            print(w)

        weigth_candidates = np.array(weigth_candidates)

        weigth_occurances = weigth_candidates.mean(axis=0)

        index_of_high_occurances = np.where(weigth_occurances > 0.45)

        combined = np.zeros_like(weigth_occurances)

        combined[index_of_high_occurances] = 1

        self.classfier_weigths = combined

        print("weigths: ", np.round(self.classfier_weigths, 3))
        self.update_mask()
        print()
        print()


    def update_mask(self):
        # TODO rework needed after networks appended
        obsolete_classifiers_indexes = np.where(self.classfier_weigths == 0)[0]
        index_obsolete = 0
        for index in range(len(self.simplemodels.classifiers_mask)):
            if self.simplemodels.classifiers_mask[index]:
                if index_obsolete in obsolete_classifiers_indexes:
                    self.simplemodels.classifiers_mask[index] = 0
                index_obsolete += 1
        self.classfier_weigths = np.delete(self.classfier_weigths, obsolete_classifiers_indexes)

