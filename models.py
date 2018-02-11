import preprocessing
import simplemodels
import numpy as np
from help_functions import Optimize_classifier_weigths, reshape_x
import neuro_networks


class main_model:
    def __init__(self, preprocess_class):
        self.preprocess_class = preprocess_class

        self.simplemodels = simplemodels.simplemodels(1, preprocess_class)
        self.networks = neuro_networks.neuronetwork_models(preprocess_class)

        self.classfier_weigths = None


    def test_full(self):
        print("###########################################################")
        print("training models to test")
        self.simplemodels.mode = 1
        simple_model_predicts = self.simplemodels.all_simple_models()

        simple_model_predicts = reshape_x(simple_model_predicts)

        y,y_test = self.preprocess_class.get_labels()

        if self.classfier_weigths is None:
            self.classfier_weigths  =np.ones(simple_model_predicts.shape[1])

        print("weigths: ", np.round(self.classfier_weigths, 3))

        weigthed_predicts = np.argmax(np.dot(self.classfier_weigths, simple_model_predicts), axis=1)

        acc = np.mean(weigthed_predicts == y_test)

        print ("test accuracy: ", acc)

        return


    def get_submissions(self):
        print("###########################################################")
        print("training models to make submission")
        self.simplemodels.mode = 3
        simple_model_predicts = self.simplemodels.all_simple_models()

        simple_model_predicts = reshape_x(simple_model_predicts)

        y,_ = self.preprocess_class.get_labels()

        if self.classfier_weigths is None:
            self.classfier_weigths  = np.ones(simple_model_predicts.shape[1])

        print("nonzero weigths: ", np.round(self.classfier_weigths, 3))





        weigthed_submission = np.argmax(np.dot(self.classfier_weigths, simple_model_predicts), axis=1)

        print("submission shape: ", weigthed_submission.shape)
        return weigthed_submission

    def train_classifier_weigths(self):
        print ("###########################################################")
        print ("training models to optimize classifier weigths")
        self.simplemodels.mode = 1
        predicts = self.simplemodels.all_simple_models()
        y_test, y_train = self.preprocess_class.get_labels()

        #TODO add test data into predicts

        #TODO how weigths would come if trained on train+test data
        optimizer = Optimize_classifier_weigths()
        self.classfier_weigths = optimizer.train_theta(predicts, y_train)#, x_test, y_test)




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










