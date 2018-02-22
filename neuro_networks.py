import numpy as np
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
import preprocessing
from sklearn.metrics import accuracy_score
from keras import backend as K


from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, concatenate, add, Activation
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras import regularizers
from sklearn import svm
import os.path

import os

class neuronetwork_models:
    def __init__(self, preprocess_class):
        self.preprocess_class = preprocess_class
        self.train_sets = 1 #how many times train
        self.is_submission = False


    def all_neuronets(self):
        print("neuro models")
        all_predicts = self.loop_neuronets()
        all_predicts = np.transpose(np.array(all_predicts), (1, 0, 2))
        K.clear_session() # clears so doesn't overflow if training multiple networks one after another
        return all_predicts


    def external_probas(self):
        return # looppaaa ja palauttaa

    def loop_neuronets(self):
        model_name = "convtestmodel32.h5"
        x_train, y_train_labels, x_test, y_test_labels = self.get_train_test_data()

        y_train = to_categorical(y_train_labels)

        c = 0.0001
        inputTensor_1 = Input((40, 501, 1))
        layer_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputTensor_1)
        layer_1 = MaxPooling2D(pool_size=(3, 3))(layer_1)
        layer_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(layer_1)
        layer_1 = MaxPooling2D(pool_size=(3, 8))(layer_1)

        layer_1 = Flatten()(layer_1)
        layer_1 = Dropout(.7)(layer_1)

        layer_1 = Dense(25, activation='relu',
                        kernel_regularizer=regularizers.l1(c),
                        activity_regularizer=regularizers.l1(c))(layer_1)

        main_output = Dense(15, activation='sigmoid',
                            kernel_regularizer=regularizers.l1(c),
                            activity_regularizer=regularizers.l1(c))(layer_1)

        model = Model(inputs=inputTensor_1, outputs=main_output)

        #print(model.summary())
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

        print("model name: ", model_name, end=" ")

        if os.path.isfile(model_name):
            model.load_weights(model_name)
            print("model loaded")
        else:
            print("model weigths not found, starting from scratch")

        if self.train_sets>0:
            sets = self.train_sets
            print("training model",sets,"epochs")
            for n in range(sets):
                model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=0)
                predict_proba = model.predict(x_train)
                predict = np.argmax(predict_proba, axis=1)
                accuracy = accuracy_score(y_train_labels, predict)

                print("progress: %1d/%1d, train accuracy: %.3f" % (n+1, sets,accuracy), end=" ")
                if y_test_labels is not None:  # test accuracy if can
                    predict_proba = model.predict(x_test)
                    predict = np.argmax(predict_proba, axis=1)
                    accuracy_test = accuracy_score(y_test_labels, predict)
                    print("test accuracy: %.3f " % (accuracy_test), end="")
                print()


                model.save(model_name)

        predict_proba = model.predict(x_test)

        return [predict_proba]

    def get_train_test_data(self):
        y_test = None
        if self.is_submission:
            self.preprocess_class.is_submission = True
            X_train, X_test = self.preprocess_class.full_image()
            y_train, _ = self.preprocess_class.get_labels()
        else:
            X_train, X_test = self.preprocess_class.full_image()
            y_train, y_test = self.preprocess_class.get_labels()

        # semisv doesn't change unless initialised
        X_train, y_train = self.preprocess_class.semisupervised_data(X_train, y_train, X_test)

        return X_train, y_train, X_test, y_test