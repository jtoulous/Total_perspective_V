import mne
import pandas as pd
from colorama import Fore, Style
from joblib import load
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from utils.tools import printLog, printError
from utils.preprocessing import getLabel, myPCA, getAlphaSignals, getBetaSignals, getGammaSignals


def EDFPrediction(pipeline):
    while True:
        file_path = input(Fore.GREEN + '\nPrecise the path to the edf file you wish to playback(or \'done\' to exit):\n==> ' + Style.RESET_ALL)
        if file_path == 'done':
            break
        file_name = file_path.split('/')[-1]
        try :
            raw = mne.io.read_raw_edf(file_path, preload=True)
        except Exception:
            printError('error: Incorrect file')
            continue

        annotations = raw.annotations
        good_pred = 0
        total_pred = 0
        printLog(f'\n=========> Subject {file_name}:')
        for a, annotation in enumerate(annotations):
            total_pred += 1
            new_data = {}
            start = int(annotation['onset'] * raw.info['sfreq'])
            end = start + int(annotation['duration'] * raw.info['sfreq'])
            segment = raw[:, start:end]

            new_data = getAlphaSignals(new_data, segment, raw.info['sfreq'])
            new_data = getBetaSignals(new_data, segment, raw.info['sfreq'])
            new_data = getGammaSignals(new_data, segment, raw.info['sfreq'])
            tmp_df = pd.DataFrame([new_data])
            prediction = pipeline.predict(tmp_df)
                
            label = getLabel(file_name, annotation['description'])
            if label == prediction[0]:
                good_pred += 1
                printLog(f'- experiment {a + 1}: ' + label + f' ===> {prediction[0]}')
            else:
                printError(f'- experiment {a + 1}: ' + label + f' ===> {prediction[0]}')
        printLog(f'Accuracy: {(good_pred / total_pred)}')


def CSVPrediction(pipeline):
    file_path = input(Fore.GREEN + '\nPrecise the path to the csv file you wish to use:\n==> ' + Style.RESET_ALL)
    true_labels = None
    try:
        X = pd.read_csv(file_path)
    except Exception:
        printError('error: Incorrect file')
        return

    if 'label' in list(X.columns):
        true_labels = X['label']
        X = X.drop('label', axis=1)
    if 'id' in list(X.columns):
        X = X.drop('id', axis=1)

    predictions = pipeline.predict(X)
    if true_labels is None:
        for prediction in predictions:
            printLog(prediction)
    else:
        accuracy = accuracy_score(true_labels, predictions)
        printLog(f'Accuracy on test set: {accuracy * 100:.2f}%')
        printLog(classification_report(true_labels, predictions))



if __name__ == "__main__":
    try:
        pipeline = load('data/pipeline.joblib')
        while True:
            usage = input(Fore.GREEN + '\nPick a number:\n  1- EDF file\n  2- CSV file\n  3- Done\n==> ' + Style.RESET_ALL).strip()
            if usage == '1':
                EDFPrediction(pipeline)
            elif usage == '2':
                CSVPrediction(pipeline)
            elif usage == '3':
                break
            else: 
                printError('error: Choice must be one of the values above')

    except Exception as error:
        printError(str(error))