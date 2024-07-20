import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from joblib import dump

from utils.tools import printError, printLog
from utils.preprocessing import getData, OverSample, myPCA

import numpy as np


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=300, help='number of epochs')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse()

        printLog('====> Reading data...')
        dataframe, features = getData()
        X = dataframe[features]
        y = dataframe['label']
        printLog('====> Done')

        printLog('====> Over sampling...')
        X, y = OverSample(X, y)
        printLog('====> Done')

#        printLog('====> Splitting data...')
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#        printLog('====> Done')

        printLog('====> Creating pipelines...')
        preprocessing = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', myPCA(n_components=180))
        ])

        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('mlp', MLPClassifier(random_state=42, max_iter=args.epochs, activation='logistic', verbose=True))
        ])
        printLog('====> Done')


        printLog('=====> Performing cross-validation...')
        accuracy_scorer = make_scorer(accuracy_score)
        scores = cross_val_score(pipeline, X, y, cv=3, scoring=accuracy_scorer)
        printLog('Cross-validation scores: {}'.format(scores))
        printLog('Mean cross-validation score: {:.2f}%'.format(scores.mean() * 100))
        printLog('=====> Done')


        printLog('====> Training model...')
        pipeline.fit(X, y)
        printLog('====> Done')

        printLog('====> Predicting...')
        predictions = pipeline.predict(X)
        printLog('====> Done')

        accuracy = accuracy_score(y, predictions)
        printLog(f'Accuracy on test set: {accuracy * 100:.2f}%')
        printLog(classification_report(y, predictions))
        dump(pipeline, 'data/pipeline.joblib')

    except Exception as error:
        printError(f'error: {error}')