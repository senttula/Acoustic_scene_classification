for competition https://www.kaggle.com/c/acoustic-scene-2018/

Idea: use multiple classifiers and combine the probabilities for each item with some weigths. Each classifier can preprocess the data differently. Loop everything over all cross validation folds to generalize.

Give each classifier weigth by optimizing with backward/forward recursive choosing (other methods weren't that efficent). Choose weigths that appeared the most over crossvalidation folds. the forward recursive choosing usually was better meaning a few classifiers is enough.

Semisupervision: after training and combining all, append labels that had enough probability and confidence. This functions input and output are predict probabilities meaning it can be looped. Parameters tested to be best: min probablity 0.5, confidence (no other probability is over this) 0.3, how many times loop the semisupervision: 3 and not overwriting predicts made by previous semisupervision loop. Neuronets weren't teached multiple times.

accuracy progress with new techniques:

72% using multiple classifiers with same weigths and data averaged by time

75% added new features of data: deviation, median, min, max, skew, kurtosis for each frequency. Kurtosis and skewness turned out to be confusers as they were time dependent.

77% added semisupervised learning

78% added confidence threshold to semisupervision

79% looped the semisupervision more

79.5% made semisupervised learning not to overwrite its previous predictions

80% combined with larger network by Jorma Syrjä

Problems faced: 

Every simple classifier returns normalized probabilities for each item, while neuronetworks are usually normalized over all items.

How acccuracy could be improved:

A single convolutional network should be able to achieve close to same accuracies but best ones made were around 73%. Optimizing the size of the network would be next step, with more computational power and time.

Simple features, such as mean and deviation of all frequencys, achieve decent results, but are some frequencys unnecessary and are there more features that could be calculated from audiospectrum?



