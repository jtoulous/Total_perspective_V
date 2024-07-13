from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

from utils.tools import printError, printLog
from utils.preprocessing import getData, UnderSample


if __name__ == '__main__':
    try:
        printLog('TRAINING START:')

        printLog('====> Reading data...')
        dataframe, features = getData()
        X = dataframe[features]
        y = dataframe['label']
        printLog('====> Done')

#        printLog('====> Under sampling...')
#        X, y = UnderSample(X, y)
#        breakpoint()
#        printLog('====> Done')

        printLog('====> Over sampling...')
        sampler = RandomOverSampler(random_state=42)
        X, y = sampler.fit_resample(X, y)
        printLog('====> Done')

        printLog('====> Splitting data...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        printLog('====> Done')

        printLog('====> Creating pipelines...')
        preprocessing = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=150))
        ])

        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('mlp', MLPClassifier(random_state=42, max_iter=1000, activation='logistic', verbose=True))
        ])
        printLog('====> Done')

        printLog('=====> Training model...')
        pipeline.fit(X_train, y_train)
        printLog(' =====> Done')

        printLog('=====> Predicting...')
        predictions = pipeline.predict(X_test)
        printLog('=====> Done')

        y_test.reset_index(drop=True, inplace=True)
        good_pred = 0
        for p, prediction in enumerate(predictions):
            if prediction == y_test[p]:
                good_pred += 1
                printLog(f'{y_test[p]} ===> {prediction}')
            else:
                printError(f'{y_test[p]} ===> {prediction}')

        printLog(f'{(good_pred / len(predictions) * 100)}% correct predictions')
        printLog('DONE')

        accuracy = accuracy_score(y_test, predictions)
        printLog(f'Accuracy on test set: {accuracy * 100:.2f}%')
        printLog(classification_report(y_test, predictions))

    except Exception as error:
        printError(f'error: {error}')