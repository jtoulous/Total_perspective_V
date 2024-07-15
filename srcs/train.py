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



if __name__ == '__main__':
    try:
        printLog('TRAINING START:')

        printLog('====> Reading data...')
        dataframe, features = getData()
        X = dataframe[features]
        y = dataframe['label']
        printLog('====> Done')

        printLog('====> Over sampling...')
        X, y = OverSample(X, y)
        printLog('====> Done')

        printLog('====> Splitting data...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        printLog('====> Done')

        printLog('====> Creating pipelines...')
        preprocessing = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', myPCA(n_components=150))
        ])

        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('mlp', MLPClassifier(random_state=42, max_iter=10, activation='logistic', verbose=True))
        ])
        printLog('====> Done')


        printLog('=====> Performing cross-validation...')
        accuracy_scorer = make_scorer(accuracy_score)
        scores = cross_val_score(pipeline, X, y, cv=5, scoring=accuracy_scorer)
        printLog('Cross-validation scores: {}'.format(scores))
        printLog('Mean cross-validation score: {:.2f}%'.format(scores.mean() * 100))
        printLog('=====> Done')


        printLog('=====> Training model...')
        pipeline.fit(X_train, y_train)
        printLog(' =====> Done')

        printLog('=====> Predicting...')
        predictions = pipeline.predict(X_test)
        printLog('=====> Done')

        accuracy = accuracy_score(y_test, predictions)
        printLog(f'Accuracy on test set: {accuracy * 100:.2f}%')
        printLog(classification_report(y_test, predictions))
        dump(pipeline, 'data/pipeline.joblib')

    except Exception as error:
        printError(f'error: {error}')