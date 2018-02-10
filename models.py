import preprocessing
import simplemodels
import numpy as np
from help_functions import Optimize_classifier_weigths, reshape_x


class main_model:
    def __init__(self, preprocess_class):

        self.preprocess_class = preprocess_class

        self.simplemodels = simplemodels.simplemodels(1, preprocess_class)


        self.classfier_weigths = None


    def test_full(self):
        print("###########################################################")
        print("training models to test")
        self.simplemodels.mode = 3
        submission_predicts = self.simplemodels.all_simple_models()

        submission_predicts = reshape_x(submission_predicts)

        y,_ = self.preprocess_class.get_labels()

        if self.classfier_weigths is None:
            self.classfier_weigths  =np.ones(submission_predicts.shape[1])

        print("weigths: ", np.round(self.classfier_weigths, 3))

        weigthed_submission = np.argmax(np.dot(self.classfier_weigths, submission_predicts), axis=1)

        print("submission shape: ", weigthed_submission.shape)
        return weigthed_submission





    def get_submissions(self):
        print("###########################################################")
        print("training models to make submission")
        self.simplemodels.mode = 3
        submission_predicts = self.simplemodels.all_simple_models()

        submission_predicts = reshape_x(submission_predicts)

        y,_ = self.preprocess_class.get_labels()

        if self.classfier_weigths is None:
            self.classfier_weigths  =np.ones(submission_predicts.shape[1])

        print("nonzero weigths: ", np.round(self.classfier_weigths, 3))

        weigthed_submission = np.argmax(np.dot(self.classfier_weigths, submission_predicts), axis=1)

        print("submission shape: ", weigthed_submission.shape)
        return weigthed_submission


    def train_classifier_weigths(self):
        print ("###########################################################")
        print ("training models to optimize classifier weigths")
        self.simplemodels.mode = 1
        predicts = self.simplemodels.all_simple_models()
        _, y = self.preprocess_class.get_labels()
        optimizer = Optimize_classifier_weigths()
        self.classfier_weigths = optimizer.train_theta(predicts, y)
        print("weigths: ",np.round(self.classfier_weigths, 3))

        obsolete_classifiers_indexes  =np.where(self.classfier_weigths == 0)[0]
        for index in obsolete_classifiers_indexes:
            print("weigth zero, skipping: ",self.simplemodels.classifiers[index][2])
            self.simplemodels.classifiers_mask[index] = False

        self.classfier_weigths = np.delete(self.classfier_weigths, obsolete_classifiers_indexes)


        print()
        print()










