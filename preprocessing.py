import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from time import time
from random import shuffle


class preprocess:
    #holds the data and returns with wanted preprocess
    def __init__(self, is_submission=False):
        #TODO this initializing takes long, find if can be faster
        self.is_submission = is_submission
        self.label_encoder = LabelEncoder()
        self.features_by_frequency_list = []

        self.read_data()

        # crossvalidation
        self.nfold=5
        self.split_crossvalidation_metatrain(self.nfold)
        self.change_crossvalidation()

        # semisupervised learning
        self.threshold = 0.5
        self.confidence_threshold = 0.3
        self.semisvdata = None  # tuple: x indexes, y indexes


    def read_data(self):
        try:
            self.x_train_file = np.load("X_train.npy", mmap_mode="r")
            self.y_train_file = np.loadtxt("y_train.csv", dtype=str, skiprows=1, delimiter=",", usecols=1)
            self.X_submission = np.load("X_test.npy", mmap_mode="r")
            self.label_encoder.fit(list(set(self.y_train_file)))
        except:
            raise Exception("missing files?")


    def split_crossvalidation_metatrain(self, nfold):
        self.label_encoder.fit(list(set(self.y_train_file)))
        self.y_train_encoded = self.label_encoder.transform(self.y_train_file)

        meta_train_file = open("meta_train.csv")
        meta_rows = []
        for line in meta_train_file:
            if line.startswith("id"):
                continue
            line = line.split(",")
            line[2] = line[2].strip()
            meta_rows.append(line)

        all = {}

        for index in range(len(meta_rows)):
            id = meta_rows[index][1]
            label = meta_rows[index][2]
            if label in all:
                if id in all[label]:
                    all[label][id].append(index)
                else:
                    all[label][id] = [index]
            else:
                all[label] = {id: [index]}

        self.meta_indexes = [[] for _ in range(nfold)]

        iter = 0
        for key, value in all.items():
            for key, value in value.items():
                for i in value:
                    self.meta_indexes[iter].append(i)
                iter += 1
                if iter >= nfold:
                    iter = 0
        self.nfold=nfold

    def change_crossvalidation(self, crossvalidation_fold_number=3):

        test_indexes = self.meta_indexes[crossvalidation_fold_number]

        # this line is trainwreck but takes other indexes that are not on test_indexes
        train_indexes = [a for i, sub_lists in enumerate(self.meta_indexes) if i != crossvalidation_fold_number
                         for a in sub_lists]

        shuffle(test_indexes)
        shuffle(train_indexes)

        self.inputtrain  = np.array(self.x_train_file[train_indexes])
        self.outputtrain = np.array(self.y_train_encoded[train_indexes])

        self.inputtest  = np.array(self.x_train_file[test_indexes])
        self.outputtest = np.array(self.y_train_encoded[test_indexes])

        #print(self.inputtrain .shape)
        #print(self.outputtrain.shape)
        #print(self.inputtest  .shape)
        #print(self.outputtest .shape)

        self.init_features_by_frequency()



    def init_features_by_frequency(self):
        # initilasing this takes few seconds but saves if many classificators are used
        self.features_by_frequency_list = []
        # takes few seconds but saves when when calling data
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
        # [np.mean, np.std, stats.skew, stats.kurtosis, np.median, np.min, np.max]

        X_train_all = self.features_by_frequency_list[0]
        X_test_all = self.features_by_frequency_list[1]
        X_submission_all =self.features_by_frequency_list[2]

        X_train = []
        X_test = []

        for i in range(7):
            if mask[i]:
                X_train.append(X_train_all[i])
                X_test.append(X_test_all[i])

        X_train = np.transpose(X_train, (1, 0, 2))
        X_test = np.transpose(X_test, (1, 0, 2))
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))

        if self.is_submission:
            X_submission = []
            for i in range(7):
                if mask[i]:
                    X_submission.append(X_submission_all[i])
            X_submission = np.transpose(X_submission, (1, 0, 2))
            X_submission = X_submission.reshape((X_submission.shape[0], -1))
            return np.concatenate((X_train, X_test)), X_submission
        else:
            return X_train, X_test

    def full_image(self):
        X_train = self.inputtrain
        X_test = self.inputtest

        X_train = X_train[:,:,:,np.newaxis]
        X_test = X_test[:,:,:, np.newaxis]

        if self.is_submission:
            X_submission = self.X_submission
            X_submission = X_submission[:, :, :, np.newaxis]
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



    def semisupervised_data(self, x, y, x_submission):

        #TODO better distinguish on the best one (0.5,0.1,0.1,0.1,0.1) is sure but
        #TODO (0.5,0.46,0.01,0.01,0.01) is not so sure but with the same threshold

        if self.semisvdata is None:
            return x, y

        y = y.reshape(-1, 1)

        new_x = x_submission[self.semisvdata[0]]
        new_y = self.semisvdata[1].reshape(-1, 1)

        new_train_x_all = np.concatenate((x,new_x))
        new_train_y_all = np.concatenate((y,new_y))

        return new_train_x_all, new_train_y_all.ravel()

    def init_semisupervised(self, probas):
        """
        adds best predicts to train data
        adds if probability > threshold and no other probability is over confidence_threshold

        might not be fastest solution but this initialising isn't called often
        """
        new_set = np.where(probas > self.threshold)
        if self.confidence_threshold == 0 or self.threshold+self.confidence_threshold>1:
            self.semisvdata = new_set

        index_high_threshold = new_set[0]
        labels_high_threshold = new_set[1]

        new_set = np.where(probas[index_high_threshold] > self.confidence_threshold)

        index_all_low_threshold = new_set[0]

        uniques, counts_low_threshold = np.unique(index_all_low_threshold, return_counts=True)

        non_multiples = np.where(counts_low_threshold == 1)

        confident_low_threshold_indexes = uniques[non_multiples]

        new_indexes = index_high_threshold[confident_low_threshold_indexes]
        new_labels = labels_high_threshold[confident_low_threshold_indexes]

        self.semisvdata = (new_indexes,new_labels)
        print("%.2f%% of test data added as semisupervised"% (len(new_indexes)/len(probas)))


    def reset_semisupervised(self):
        self.semisvdata = None
