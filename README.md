# music-categorizer

for competition https://www.kaggle.com/c/acoustic-scene-2018/

Loop over all cross validation folds to generalize
Predict probabilities with different classifiers.
Give each classifier weigth by optimizing with gradient descent (with a few different loss functions), test also 0/1 weigths for each classifier. Take weigths that had best accuracy.

(Optional) learn best semi-supervised learning threshold by looping all cross validation folds with different thresholds and calculate average accuracy change for each


When we have the best overall classifier weigths and semi-supervised learning threshold we can train with full data, then with the semisupervised and make the submission.
