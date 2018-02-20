# music-categorizer

for competition https://www.kaggle.com/c/acoustic-scene-2018/

Loop over all cross validation folds to generalize
Predict probabilities with different classifiers.

Every simple classifier returns normalized probabilities for each item, while neuronetworks are normalized over all.

Give each classifier weigth by optimizing with backward/forward recursive choosing (other methods weren't that efficent). Choose weigths that appeared the most over crossvalidation folds. Non-linearity when adding the probabilities wasn't helpful

With full data: train all classifiers, train again by appending the labels that had enough probability and confidence (semisupervised learning), loop the semisupervised learning few times

When we have the best overall classifier weigths and semi-supervised learning threshold we can train with full data, then with the semisupervised and make the submission.

accuracy progress with new techniques:
72% using multiple classifiers with same weigths and data averaged by time
75% added new features of data: deviation, median, min, max, skew, kurtosis for each frequency
77% added semisupervised learning
78% added confidence threshold to semisupervision
79% looped the semisupervision more