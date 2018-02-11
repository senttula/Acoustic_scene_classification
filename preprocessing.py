import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from help_functions import reshape_x

from time import  time

class preprocess:
    #holds the data and returns with wanted preprocess
    def __init__(self, is_submission=False):
        self.is_submission = is_submission
        self.label_encoder = LabelEncoder()
        self.read_data()

        self.features_by_frequency_list = []
        self.init_features_by_frequency()


    def read_data(self):
        try:
            self.inputtrain = np.load("Xtrain.npy", mmap_mode="r")
            self.outputtrain = np.load("ytrain.npy", mmap_mode="r")
            self.inputtest = np.load("Xtest.npy", mmap_mode="r")
            self.outputtest = np.load("ytest.npy", mmap_mode="r")
            self.X_submission = np.load("X_test.npy", mmap_mode="r")
            y_train = np.loadtxt("y_train.csv", dtype=str, skiprows=1, delimiter=",", usecols=1)
            self.label_encoder.fit(list(set(y_train)))
        except:
            print ("reading and splitting data")
        try:
            crossvalidation_file = open("crossvalidation_train.csv")
            self.X_submission = np.load("X_test.npy", mmap_mode="r")
            X_train = np.load("X_train.npy", mmap_mode="r")
            y_train = np.loadtxt("y_train.csv", dtype=str, skiprows=1, delimiter=",", usecols=1)
        except:
            raise Exception("missing files")
        crossvalidation_rows = []
        for line in crossvalidation_file:
            if line.startswith("id"):
                continue
            crossvalidation_rows.append(line.split(",")[2].strip())


        self.label_encoder.fit(list(set(y_train)))
        y_train_encoded = self.label_encoder.transform(y_train)

        x_tr = []
        y_tr = []

        x_tst = []
        y_tst = []

        for index in np.arange(np.size(X_train, 0)):
            if crossvalidation_rows[index] == "train":
                x_tr.append(X_train[index])
                y_tr.append(y_train_encoded[index])
            else:
                x_tst.append(X_train[index])
                y_tst.append(y_train_encoded[index])

        np.save("Xtrain.npy",np.array(x_tr))
        np.save("ytrain.npy",np.array(y_tr))
        np.save("Xtest.npy", np.array(x_tst))
        np.save("ytest.npy", np.array(y_tst))
        self.inputtrain = np.array(x_tr)
        self.outputtrain = np.array(y_tr)
        self.inputtest = np.array(x_tst)
        self.outputtest = np.array(y_tst)

    def init_features_by_frequency(self):
        features = [np.mean, np.std, stats.skew, stats.kurtosis, np.median, np.min, np.max]

        X_train = []
        X_test = []
        X_submission = []
        for i in range(7):
            X_train.append(features[i](self.inputtrain, axis=2))
            X_test.append(features[i](self.inputtest, axis=2))
            X_submission.append(features[i](self.X_submission, axis=2))
        self.features_by_frequency_list.append(X_train)
        self.features_by_frequency_list.append(X_test)
        self.features_by_frequency_list.append(X_submission)




    def mean_time(self):
        #replaced by features_by_frequency
        return self.features_by_frequency([1,0,0,0,0,0,0])

    def features_by_frequency(self, mask):
        # mask takes wanted features from features list
        #[np.mean, np.std, stats.skew, stats.kurtosis, np.median, np.min, np.max]

        X_train_all = self.features_by_frequency_list[0]
        X_test_all = self.features_by_frequency_list[1]
        X_submission_all =self.features_by_frequency_list[2]

        X_train = []
        X_test = []

        for i in range(7):
            if mask[i]:
                X_train.append(X_train_all[i])
                X_test.append(X_test_all[i])

        X_train = reshape_x(np.array(X_train))
        X_test = reshape_x(np.array(X_test))
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))

        if self.is_submission:
            X_submission = []
            for i in range(7):
                if mask[i]:
                    X_submission.append(X_submission_all[i])
            X_submission = reshape_x(np.array(X_submission))
            X_submission = X_submission.reshape((X_submission.shape[0], -1))
            return np.concatenate((X_train, X_test)), X_submission
        else:

            return X_train, X_test

    def full_image(self):
        X_train = self.inputtrain
        X_test = self.inputtest

        if self.is_submission:
            X_submission = self.X_submission
            return np.concatenate((X_train, X_test)), X_submission #TODO axis?
        else:
            return X_train, X_test

    def get_labels(self):
        y_train = self.outputtrain
        y_test = self.outputtest

        if self.is_submission:
            submission_labels = None #no labels here
            return np.concatenate((y_train, y_test)), submission_labels
        else:
            return y_train, y_test






