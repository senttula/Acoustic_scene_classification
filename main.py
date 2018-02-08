import preprocessing
import models

# Tuukka Senttula
# https://github.com/senttula

# https://www.kaggle.com/c/acoustic-scene-2018/


def make_submission_file(submission_predictions):
    print("making submission...")
    prediction_labels = list(preprocess_class.label_encoder.inverse_transform(submission_predictions))
    with open("submission.csv", "w") as fp:
        fp.write("Id,Scene_label\n")
        for i, label in enumerate(prediction_labels):
            fp.write("%d,%s\n" % (i, label))
    print("submission saved: submission.csv")

if __name__ == "__main__":
    preprocess_class = preprocessing.preprocess()
    mdl = models.main_model(preprocess_class)

    submission_predictions = mdl.get_submissions()

    make_submission_file(submission_predictions)













