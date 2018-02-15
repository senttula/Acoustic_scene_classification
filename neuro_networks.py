import numpy as np
from matplotlib.image import imread
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, concatenate, add
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.utils import to_categorical
from sklearn import svm



import os

class neuronetwork_models:
    def __init__(self, preprocess_class):
        self.preprocess_class = preprocess_class
        self.classifiers = """
        NORMAL 7
        convolution (2?)
        recurrent
        conv+recurrent
        """


    def all_neuronetwork_models(self):

        # TODO everything in progress

        x_train, x_test = self.preprocess_class.full_image()
        fx_train, fx_test = self.preprocess_class.features_by_frequency([1,1,0,0,1,1,0])

        y_train, y_test = self.preprocess_class.get_labels()



        #clf = svm.SVC(kernel="rbf", probability=True, gamma=.0172)
        #clf.fit(fx_train, y_train)
#
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
#
        #predict_proba = clf.predict_proba(fx_test)

        #model = Sequential()
        #model.add(Conv2D(10, (7,5), input_shape=(40, 501, 1), activation='relu',
        #                 padding='same'))
        #model.add(MaxPooling2D(pool_size=(4, 12)))
#
        #model.add(Conv2D(10, (7,5), activation='relu',
        #                 padding='same'))
        #model.add(MaxPooling2D(pool_size=(5, 10)))
#
        #model.add(Flatten())


        inputTensor_1 = Input((40, 501, 1))
        layer_1 = Conv2D(10, (7, 5), activation='relu',padding='same')(inputTensor_1)
        layer_1 = MaxPooling2D(pool_size=(4, 12))(layer_1)
        layer_1 = Conv2D(8, (7, 5), activation='relu',padding='same')(layer_1)
        layer_1 = MaxPooling2D(pool_size=(5, 10))(layer_1)
        layer_1 = Flatten()(layer_1)
        layer_1 = Dense(15, activation='sigmoid')(layer_1)


        inputTensor_2 = Input((160,))
        layer_2 = Dense(200, activation='relu')(inputTensor_2)
        layer_2 = Dense(50, activation='relu')(layer_2)

        layer_2 = Dense(15, activation='sigmoid')(layer_2)

        #x = concatenate([layer_1, layer_2])

        main_output = add([layer_1, layer_2])

        #x = Dense(64, activation='relu')(x)

        #main_output = Dense(15, activation='softmax', name='main_output')(x)

        model = Model(inputs=[inputTensor_1, inputTensor_2], outputs=main_output)

        print(model.summary())
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

        print(x_train.shape)
        print(fx_test.shape)
        print(x_test.shape)
        print(fx_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        #model.load_weights("convtestmodel.h5")
        for n in range(200):
            print("##########################", n)
            model.fit([x_train, fx_train], y_train, epochs=10, batch_size=64)

            score = model.evaluate([x_test, fx_test], y_test, batch_size=64)
            print(score[1])
            model.save("convtestmodel.h5")


        """

        prev_score = 0
        for n in range(200):
            # model.load_weights("convtestmodel.h5")
            print("##########################", n)
            model.fit(x_train, y_train, batch_size=128, epochs=5)
            # train_score = model.evaluate(x_train, y_train, batch_size=128)
            score = model.evaluate(x_test, y_test, batch_size=32)
            print(score[1])

            model.save("convtestmodel.h5")
            # if train_score[1] >0.995:
            #    break
            # elif (prev_score) > train_score[1]:
            #    break

            # prev_score = train_score[1]





        #model.load_weights('convtestmodel.h5')


        scores = []

        a = 1
        b = 1

        for mask in [(13, 5)]:#,(9, 9), (3, 3), (5, 15),  (15,15)]:
            for mn in [15]:#,, 20 30, 25):
                for pool in [(4, 12)]:#, (4, 8), (4, 4)]:
                    score = [99, 0]

                    #try:
                    model = Sequential()
                    model.add(Conv2D(mn, mask, input_shape=(40, 501, 1), activation='relu',
                                     padding='same'))
                    model.add(MaxPooling2D(pool_size=pool))

                    model.add(Conv2D(mn, mask, activation='relu',
                                     padding='same'))

                    model.add(MaxPooling2D(pool_size=pool))

                    model.add(Flatten())
                    model.add(Dense(15, activation='softmax'))
                    print(model.summary())
                    model.compile(loss='categorical_crossentropy',
                                  optimizer='sgd',
                                  metrics=['accuracy'])

                    prev_score = 0
                    for n in range(200):
                        #model.load_weights("convtestmodel.h5")
                        print("##########################", n)
                        model.fit(x_train, y_train, batch_size=128, epochs=5)
                        #train_score = model.evaluate(x_train, y_train, batch_size=128)
                        score = model.evaluate(x_test, y_test, batch_size=32)
                        print (score[1])

                        model.save("convtestmodel.h5")
                        #if train_score[1] >0.995:
                        #    break
                        #elif (prev_score) > train_score[1]:
                        #    break

                        #prev_score = train_score[1]
                    score = model.evaluate(x_test, y_test, batch_size=32)


                    #except:pass

                    scores.append([pool, mask, mn, score[1]])
                    print(scores)

        print(score)



        model.add(Conv2D(32, (w, h), input_shape=(64, 64, 3), activation='relu',
                         padding='same'))
        model.add(MaxPooling2D(pool_size=(4, 4)))

        model.add(Conv2D(32, (w, h), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(4, 4)))

        model.add(Flatten())
        model.add(Dense(100, activation='sigmoid'))
        model.add(Dense(2, activation='sigmoid'))






        #model.load_weights('convtestmodel.h5')

        scores = []

        for a in [150,200, 250, 300]:
            for b in [10,20,50,70,90,110, 150]:
                print ("####", a,b)
                model = Sequential()
                model.add(Dense(a, input_dim=280, activation='sigmoid'))
                model.add(Dense(b, activation='sigmoid'))
                model.add(Dense(15, activation='sigmoid'))

                model.summary()

                model.compile(loss='categorical_crossentropy',
                              optimizer='sgd',
                              metrics=['accuracy'])

                model.fit(x_train, y_train, batch_size=16, epochs=400)
                score = model.evaluate(x_test, y_test, batch_size=16)
                scores.append([a,b,score[1]])

            print (scores)
        print(scores)

[[20, 10, 0.42857142857142855], [20, 20, 0.43912337662337664], [20, 50, 0.45941558441558439], [20, 60, 0.48457792207792205], [20, 70, 0.49025974025974028], [20, 80, 0.45454545454545453], [20, 90, 0.53652597402597402],
 [20, 100, 0.34253246753246752], [50, 10, 0.50974025974025972], [50, 20, 0.59172077922077926], [50, 50, 0.53246753246753242],
 [50, 60, 0.57061688311688308], [50, 70, 0.61120129870129869], [50, 80, 0.54788961038961037], [50, 90, 0.58522727272727271], [50, 100, 0.5357142857142857], [60, 10, 0.49512987012987014],
 [60, 20, 0.54707792207792205], [60, 50, 0.57467532467532467], [60, 60, 0.51217532467532467], [60, 70, 0.5535714285714286], [60, 80, 0.59172077922077926], [60, 90, 0.59983766233766234], [60, 100, 0.55194805194805197],
 [70, 10, 0.45698051948051949], [70, 20, 0.5933441558441559], [70, 50, 0.61120129870129869], [70, 60, 0.57305194805194803],
 [   70, 70, 0.5316558441558441], [70, 80, 0.58360389610389607],
 [70, 90, 0.57629870129870131], [70, 100, 0.5316558441558441], [80, 10, 0.50324675324675328], [80, 20, 0.57792207792207795], [80, 50, 0.57711038961038963], [80, 60, 0.52597402597402598], [80, 70, 0.56818181818181823],
 [80, 80, 0.5941558441558441], [80, 90, 0.62256493506493504], [80, 100, 0.45860389610389612], [90, 10, 0.53814935064935066],
 [90, 20, 0.6071428571428571], [90, 50, 0.56168831168831168], [90, 60, 0.6339285714285714], [90, 70, 0.55275974025974028], [90, 80, 0.53003246753246758], [90, 90, 0.59090909090909094],
 [90, 100, 0.5714285714285714], [100, 10, 0.53246753246753242], [100, 20, 0.49431818181818182], [100, 50, 0.5933441558441559], [100, 60, 0.42045454545454547], [100, 70, 0.55275974025974028], [100, 80, 0.60308441558441561],
 [100, 90, 0.61769480519480524], [100, 100, 0.50162337662337664], [200, 10, 0.56737012987012991], [200, 20, 0.60551948051948057],
 [200, 50, 0.61931818181818177], [200, 60, 0.6339285714285714], [200, 70, 0.63961038961038963], [200, 80, 0.62662337662337664], [200, 90, 0.62175324675324672], [200, 100, 0.57954545454545459]]

[[150, 10, 0.62175324675324672], [150, 20, 0.5625], [150, 50, 0.64935064935064934], [150, 70, 0.65990259740259738], [150, 90, 0.64935064935064934],
 [150, 110, 0.67370129870129869], [150, 150, 0.67775974025974028],
 [200, 10, 0.62094155844155841], [200, 20, 0.64529220779220775],
 [200, 50, 0.68181818181818177], [200, 70, 0.67451298701298701], [200, 90, 0.64610389610389607], [200, 110, 0.57629870129870131],
 [200, 150, 0.63879870129870131], [250, 10, 0.60876623376623373], [250, 20, 0.68019480519480524], [250, 50, 0.68831168831168832],
 [250, 70, 0.67207792207792205], [250, 90, 0.65422077922077926], [250, 110, 0.66883116883116878], [250, 150, 0.64529220779220775],
 [300, 10, 0.60146103896103897], [300, 20, 0.5803571428571429],
 [300, 50, 0.67938311688311692],
 [300, 70, 0.68100649350649356], [300, 90, 0.62824675324675328],
 [300, 110, 0.67938311688311692], [300, 150, 0.64366883116883122]]



 [[(2, 2), (3, 3), 10, 0], [(4, 8), (3, 3), 10, 0], [(4, 4), (3, 3), 10, 0], [(2, 2), (3, 3), 20, 0],
 [(4, 8), (3, 3), 20, 0], [(4, 4), (3, 3), 20, 0], [(2, 2), (3, 3), 30, 0], [(4, 8), (3, 3), 30, 0],
 [(4, 4), (3, 3), 30, 0], [(2, 2), (3, 3), 25, 0], [(4, 8), (3, 3), 25, 0], [(4, 4), (3, 3), 25, 0],
 [(2, 2), (9, 9), 10, 0.05844155844155844], [(4, 8), (9, 9), 10, 0.060064935064935064],
  [(4, 4), (9, 9), 10, 0.05844155844155844], [(2, 2), (9, 9), 20, 0], [(4, 8), (9, 9), 20, 0],
   [(4, 4), (9, 9), 20, 0], [(2, 2), (9, 9), 30, 0], [(4, 8), (9, 9), 30, 0], [(4, 4), (9, 9), 30, 0],
    [(2, 2), (9, 9), 25, 0], [(4, 8), (9, 9), 25, 0], [(4, 4), (9, 9), 25, 0], [(2, 2), (15, 5), 10, 0.05844155844155844],
     [(4, 8), (15, 5), 10, 0.05844155844155844], [(4, 4), (15, 5), 10, 0.14204545454545456], [(2, 2), (15, 5), 20, 0],
      [(4, 8), (15, 5), 20, 0]]


"""





