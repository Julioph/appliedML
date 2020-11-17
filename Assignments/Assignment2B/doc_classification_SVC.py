import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from SVClassifier import SVClassifier

def read_data(corpus_file):
    X = []
    Y = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            _, y, _, x = line.split(maxsplit=3)
            X.append(x.strip())
            Y.append(y)
    return X, Y

if __name__ == '__main__':
    t4 = time.time()
    # Read all the documents.
    X, Y = read_data('pa2b/data/all_sentiment_shuffled.txt')

    # Split into training and test parts.
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Set up the preprocessing steps and the classifier.
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),

        SVClassifier(n_iter=50)
    )

    #Train the classifier (adjust weights) and time it
    t0 = time.time()
    pipeline.fit(Xtrain, Ytrain, svclassifier__regularization_param=1/len(Xtrain))
    t1 = time.time()

    #Evaluate on the test set
    t2 = time.time()
    Yguess = pipeline.predict(Xtest)
    t3 = time.time()
    t5 = time.time()

    print('Training duration: {:.4f} seconds.'.format(t1 - t0))
    print('Prediction duration: {:.4f} seconds.'.format(t3 - t2))
    print('Program duration: {:.4f} seconds.\n'.format(t5 - t4))
    print('Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))
