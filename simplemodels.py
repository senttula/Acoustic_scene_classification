from sklearn.metrics import accuracy_score
from preprocessing import preprocess
import numpy as np

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

#TODO predict proba

class simplemodels:
    def __init__(self, mode, preprocess_class):
        self.preprocess = preprocess_class
        self.mode = mode
        rstate = 12345
        self.classifiers = [
            #TODO differentiate good classifiers for each class
            #TODO more classifiers with different preprocessors
            (LinearDiscriminantAnalysis(), self.preprocess.mean_time, "LinearDiscriminantAnalysis"),
            (svm.SVC(random_state=rstate, kernel="linear", probability=True, C=10), self.preprocess.mean_time, "SVM linear"),
            (svm.SVC(random_state=rstate, kernel="rbf", probability=True, C=15, gamma=.029, tol=0.1), self.preprocess.mean_time, "SVM rbf"),
            (KNeighborsClassifier(n_neighbors=19), self.preprocess.mean_time, "KNN"),
            (LogisticRegression(random_state=rstate, ), self.preprocess.mean_time, "LogisticRegression"),
            (RandomForestClassifier(random_state=rstate,n_estimators=180 ), self.preprocess.mean_time, "RandomForestClassifier"),
            (ExtraTreesClassifier(random_state=rstate, n_estimators=750), self.preprocess.mean_time, "ExtraTreesClassifier"),
            #obsolete (AdaBoostClassifier(random_state=rstate, ), self.preprocess.mean_time, "AdaBoostClassifier"),
            (GradientBoostingClassifier(random_state=rstate, n_estimators=120, max_depth=5), self.preprocess.mean_time, "GradientBoostingClassifier"),
            #XG boost TODO
        ]


    def all_simple_models(self):
        print ("training all simple_models, mode: ",self.mode)
        print("test accuracy | train accuracy, classifier name")
        all_predicts = self.loop_classifiers()
        print ("end all_simple_models")
        return np.array(all_predicts)

    def loop_classifiers(self):
        all_predicts = []
        for clf, prepros, name in self.classifiers:
            X_train, y_train, X_test, y_test =self.get_train_test_data(prepros)
            clf.fit(X_train, y_train)
            predict_proba = clf.predict_proba(X_test)
            if y_test is not None: #test accuracy if can
                predict = clf.predict(X_test)
                accuracy_test = accuracy_score(y_test, predict)
                accuracy_train = accuracy_score(y_train, clf.predict(X_train))
                print("%.3f | %.3f " % (accuracy_test, accuracy_train), end="")
            print(name)
            #if accuracy > 0.4:  # weak classifiers are distracting
            all_predicts.append(predict_proba)
        return all_predicts

    def get_train_test_data(self,preprocess):
        y_test = None
        if self.mode == 1:

            X_train, X_test = preprocess()
            y_train, y_test = self.preprocess.get_labels()
        elif self.mode == 2:
            X_test, X_train = preprocess()
            y_test, y_train = self.preprocess.get_labels()
        elif self.mode == 3:
            self.preprocess.is_submission = True
            X_train, X_test = preprocess()
            y_train, _ = self.preprocess.get_labels()
        else:
            raise Exception('class simplemodels: not supported mode: ', self.mode)
        return X_train, y_train, X_test, y_test