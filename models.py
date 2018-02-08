import preprocessing
import simplemodels
import numpy as np
from help_functions import Optimize_classifier_weigths, reshape_x


class main_model:
    def __init__(self, preprocess_class):
        self.simplemodels = simplemodels.simplemodels(1, preprocess_class)
        self.preprocess_class = preprocess_class

        self.classfier_weigths = None

    def get_submissions(self):
        print("###########################################################")
        print("training models to make submission")
        self.simplemodels.mode = 3
        submission_predicts = self.simplemodels.all_simple_models()

        submission_predicts = reshape_x(submission_predicts)

        y,_ = self.preprocess_class.get_labels()

        if self.classfier_weigths is not None:
            self.classfier_weigths  =np.ones(submission_predicts.shape[1])

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
        print()
        print()
        #[ 0.17   0.796  1.     0.031  0.147 -0.435  0.368  0.214]










