import preprocessing
import models

# Tuukka Senttula
# https://github.com/senttula

# https://www.kaggle.com/c/acoustic-scene-2018/

# TODO neuronetwork ROC, what threshold is confident

"""
configuration:

semisupervised thresholds

negative weigths bool

submission or test

nfold = 5

train weigthts/semisupervised iterations?

"""


def make_submission_file(submission_predictions):
    print("making submission...", submission_predictions.shape)
    prediction_labels = list(preprocess_class.label_encoder.inverse_transform(submission_predictions))
    with open("submission.csv", "w") as fp:
        fp.write("Id,Scene_label\n")
        for i, label in enumerate(prediction_labels):
            fp.write("%d,%s\n" % (i, label))
    print("submission saved: submission.csv")

if __name__ == "__main__":
    preprocess_class = preprocessing.preprocess(verbose_level=2)
    mdl = models.main_model(preprocess_class)


    #mdl.train_classifier_weigths()

    # mdl.test_full()


    submission_predictions = mdl.get_submissions()
    make_submission_file(submission_predictions)















