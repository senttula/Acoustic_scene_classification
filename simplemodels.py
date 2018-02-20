from sklearn.metrics import accuracy_score
import numpy as np

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

from help_functions import test_by_class
from time import  time


class simplemodels:
    def __init__(self, preprocess_class):
        self.preprocess = preprocess_class

        self.is_submission = False
        # mode1 = x_train, x_test
        # mode2 = x_test,  x_train (so just reversed for some testing)
        # mode3 = x_train+x_test, x_submission  (x_train+x_test concentated)
        # mode4 = x_train+x_test+high_probas_from_submission, x_submission  (semisupervised from submission)

        self.init_classifiers()

    def init_classifiers(self):
        rstate = 123454#12145#123454
        mn_time = [1,0,0,0,0,0,0]
        self.classifiers = [
            (svm.SVC(random_state=rstate, kernel="linear", probability=True, C=10), mn_time, "SVM linear"),
            (svm.SVC(random_state=rstate, kernel="rbf", probability=True, C=15, gamma=.029, tol=0.1),mn_time,"SVM rbf"),
            (svm.SVC(kernel="poly", C=1, gamma=0.06, tol=1e-02, probability=True),mn_time, "SVM poly"),
            (LinearDiscriminantAnalysis(), mn_time, "LinearDiscriminantAnalysis"),
            (KNeighborsClassifier(n_neighbors=19), mn_time, "19 NN"),
            (LogisticRegression(random_state=rstate, C=6,tol=1e-5), mn_time, "LogisticRegression"),
            (RandomForestClassifier(random_state=rstate, n_estimators=180), mn_time,"RandomForest"),
            (ExtraTreesClassifier(random_state=rstate, n_estimators=750), mn_time,"ExtraTrees"),
            (GradientBoostingClassifier(random_state=rstate, n_estimators=120, max_depth=5), mn_time,"GradientBoost"),
            #obsolete (AdaBoostClassifier(random_state=rstate, ), mn_time, "AdaBoostClassifier"),
#
            (svm.SVC(random_state=rstate, kernel="rbf", C=8, probability=True, gamma=.0172),
                                        [1,1,0,0,1,1,0], "SVM rbf 1100110"),
            (svm.SVC(random_state=rstate, kernel="linear", probability=True),[1, 1, 0, 0, 1, 0,0],"SVM linear 1100100"),
            (svm.SVC(random_state=rstate, kernel="poly", probability=True), [1, 1, 0, 0, 1, 1, 0], "SVM poly 1100110"),
            (LinearDiscriminantAnalysis(), [1,1,1,1,1,0,0,], "LinearDiscriminantAnalysis 1111100"),
            (KNeighborsClassifier(n_neighbors=3), [1, 1, 0, 0, 0, 0, 0], "3 NN 1100000"),
            (LogisticRegression(random_state=rstate), [1, 1, 0, 0, 0, 0, 0], "LogisticRegression 1100000"),
            (RandomForestClassifier(random_state=rstate, n_estimators=320), [1,1,0,0,1,0,1,], "RandomForest 1100101"),
            (GradientBoostingClassifier(random_state=rstate, n_estimators=120, max_depth=5),
                                        [1, 1, 0, 0, 1, 1, 0], "GradientBoost 1100110"),
###
            (svm.SVC(random_state=rstate, kernel="rbf", probability=True, gamma=0.0172),
             [1, 1, 0, 0, 1, 1, 1], "SVM rbf 1100110"),
            (svm.SVC(random_state=rstate, kernel="linear", probability=True), [1, 1, 0, 0, 1, 1, 1],
             "SVM linear 1100110"),
            (svm.SVC(random_state=rstate, kernel="poly", probability=True), [1, 1, 0, 0, 1, 1, 1], "SVM poly 1100110"),
            (LinearDiscriminantAnalysis(), [1, 1, 0, 0, 1, 1, 0, ], "LinearDiscriminantAnalysis 1111100"),
            (LogisticRegression(random_state=rstate), [1, 1, 0, 0, 1, 0, 0], "LogisticRegression 1100000"),
            (RandomForestClassifier(random_state=rstate, n_estimators=180), [1, 1, 0, 0, 1, 1, 1, ],
             "RandomForest 1100101"),

        ]
        self.reset_mask()

    def reset_mask(self):
        print("simple models mask reseted")
        self.classifiers_mask = [1 for i in range(len(self.classifiers))]
        #self.classifiers_mask = [1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        # 0., 0., 0., 1., 0.] #TODO just for test mask
        #self.classifiers_mask = [ 1. , 0. , 0. , 0. , 0.,  0.,  0.,  0.,  0. , 1. , 0. , 0. , 1.  ,1.  ,
        # 1.,  1. , 1. , 1., 0. , 1., 1.,  0. , 1.]  # TODO just for test mask
        #self.classifiers_mask = [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
        #                         1., 1., 0., 1., 0., 1., 1., 0., 0.]
        self.classifiers_mask = [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                                 0., 1., 0., 0., 0., 1., 1., 0., 0.]

    def all_simple_models(self):
        print ("training all simple_models")
        print("test accuracy | train accuracy, classifier name")
        all_predicts = self.loop_classifiers()
        print("end all_simple_models")
        all_predicts = np.transpose(np.array(all_predicts), (1,0,2))
        return all_predicts

    def loop_classifiers(self):
        all_predicts = []
        for i, clf_info in enumerate(self.classifiers):
            clf, mask, name = clf_info
            if self.classifiers_mask[i]: #skip useless classifiers
                X_train, y_train, X_test, y_test = self.get_train_test_data(mask)

                clf.fit(X_train, y_train)
                predict_proba = clf.predict_proba(X_test)
                if y_test is not None:#test accuracy if can
                    predict = clf.predict(X_test)
                    accuracy_test = accuracy_score(y_test, predict)
                    accuracy_train = accuracy_score(y_train, clf.predict(X_train))
                    print("%.3f | %.3f " % (accuracy_test, accuracy_train), end="")
                    # test_by_class(predict, y_test)
                print(name)
                # if accuracy < 0.4:  # weak classifiers are distracting
                all_predicts.append(predict_proba)
            else:
                print ("----- | ----- skipping: (", name, ")")
        return all_predicts

    def get_train_test_data(self,preprocess_mask):
        y_test = None
        if self.is_submission:
            self.preprocess.is_submission = True
            X_train, X_test = self.preprocess.features_by_frequency(preprocess_mask)
            y_train, _ = self.preprocess.get_labels()
        else:
            X_train, X_test = self.preprocess.features_by_frequency(preprocess_mask)
            y_train, y_test = self.preprocess.get_labels()

        # semisv doesn't change unless initialised
        X_train, y_train = self.preprocess.semisupervised_data(X_train, y_train, X_test)

        return X_train, y_train, X_test, y_test
