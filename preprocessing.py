import numpy as np
from sklearn.preprocessing import LabelEncoder

class preprocess:
    #holds the data and returns with wanted preprocess
    def __init__(self, is_submission=False):
        self.is_submission = is_submission
        self.label_encoder = LabelEncoder()
        try:
            self.inputtrain = np.load("Xtrain.npy", mmap_mode="r")
            self.outputtrain = np.load("ytrain.npy", mmap_mode="r")
            self.inputtest  = np.load("Xtest.npy", mmap_mode="r")
            self.outputtest = np.load("ytest.npy", mmap_mode="r")
            self.X_submission = np.load("X_test.npy", mmap_mode="r")
            y_train = np.loadtxt("y_train.csv", dtype=str, skiprows=1, delimiter=",", usecols=1)
            self.label_encoder.fit(list(set(y_train)))
        except:
            self.read()

    def read(self):
        try:
            crossvalidation_file = open("crossvalidation_train.csv")
            self.X_submission = np.load("X_test.npy", mmap_mode="r")
            X_train = np.load("X_train.npy", mmap_mode="r")
            y_train = np.loadtxt("y_train.csv", dtype=str, skiprows=1, delimiter=",", usecols=1)
        except:
            print ("missing files")
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
        np.save("Xtest.npy",np.array(x_tst))
        np.save("ytest.npy",np.array(y_tst))
        self.inputtrain = np.load("Xtrain.npy")
        self.outputtrain = np.load("ytrain.npy")
        self.inputtest = np.load("Xtest.npy")
        self.outputtest = np.load("ytest.npy")


    def to_spectrum(self):
        pass

    def mean_time(self):
        if self.is_submission:
            X_train = np.mean(self.inputtrain, axis=2)
            X_test = np.mean(self.inputtest, axis=2)
            X_submission = np.mean(self.X_submission, axis=2)
            return np.concatenate((X_train, X_test)), X_submission
        else:
            X_train = np.mean(self.inputtrain, axis=2)
            X_test  = np.mean(self.inputtest, axis=2)
            return X_train, X_test



    def get_labels(self):
        if self.is_submission:
            y_train = self.outputtrain
            y_test = self.outputtest
            return np.concatenate((y_train, y_test)), None #none submission test labels
        else:
            y_train = self.outputtrain
            y_test = self.outputtest
            return y_train, y_test






