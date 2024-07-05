from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

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
#        printLog('====> Done')

        printLog('====> Splitting data...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        printLog('====> Done')

        printLog('====> Creating pipeline...')
        preprocessing = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=10))
        ])

        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('knn', KNeighborsClassifier(n_neighbors=3))
#            ('svm', SVC(kernel='linear'))
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

    except Exception as error:
        printError(f'error: {error}')