"""
Execute the same classification algorithm but with Python scikit
library
"""
#!/usr/local/bin/python3
#pylint: disable=C0103

from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas
import numpy

if __name__ == '__main__':
    with open('dataset/SMSSpamCollection', 'r') as dataset_filename:
        data_frame = pandas.read_table(dataset_filename, sep='\t', header=None)

    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB())])

    k_fold = KFold(n=len(data_frame), n_folds=6)
    scores = []
    confusion = numpy.array([[0, 0], [0, 0]])

    for train_indices, test_indices in k_fold:
        train_text = data_frame.iloc[train_indices][1].values
        train_y = data_frame.iloc[train_indices][0].values

        test_text = data_frame.iloc[test_indices][1].values
        test_y = data_frame.iloc[test_indices][0].values

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label='spam')
        scores.append(score)

    print("Total SMS classified: {}".format(len(data_frame)))
    print("Score: {}".format(sum(scores)/len(scores)))
    print("Confusion matrix: ")
    print(confusion)
