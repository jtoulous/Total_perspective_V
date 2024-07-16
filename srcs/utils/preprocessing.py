import os
import mne
import random
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from .tools import printError, printLog


class myPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        centered_data = X - self.mean_
        cov_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        self.components_ = eigenvectors[:, -self.n_components:]
        return self

    def transform(self, X):
        centered_data = X - self.mean_
        transformed_data = np.dot(centered_data, self.components_)
        return transformed_data

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def getLabel(file, annotation_type):
    file_id = int(file.split('.')[0][-2:])
    if annotation_type == 'T0' or file_id == 1 or file_id == 2:
        return 'rest'

    if file_id == 3 or file_id == 7 or file_id == 11:
        if annotation_type == 'T1':
            return 'l_fist'
        else:
            return 'r_fist'
    
    if file_id == 4 or file_id == 8 or file_id == 12:
        if annotation_type == 'T1':
            return 'img l_fist'
        else:
            return 'img r_fist'

    if file_id == 5 or file_id == 9 or file_id == 13:
        if annotation_type == 'T1':
            return 'b_fist'
        else:
            return 'b_feet'

    if file_id == 6 or file_id == 10 or file_id == 14:
        if annotation_type == 'T1':
            return 'img b_fist'
        else:
            return 'img b_feet'


def getAlphaSignals(new_data, segment, sfreq):
    fmin, fmax = 8, 13
    for c, channel in enumerate(segment[0]):
        psd_alpha, freqs_alpha = mne.time_frequency.psd_array_multitaper(channel, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
        new_data[f'C{c} alpha'] = np.mean(psd_alpha)
    return new_data


def getBetaSignals(new_data, segment, sfreq):
    fmin, fmax = 13, 30
    for c, channel in enumerate(segment[0]):
        psd_beta, freqs_beta = mne.time_frequency.psd_array_multitaper(channel, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
        new_data[f'C{c} beta'] = np.mean(psd_beta)
    return new_data


def getGammaSignals(new_data, segment, sfreq):
    fmin, fmax = 30, 45
    for c, channel in enumerate(segment[0]):
        psd_gamma, freqs_gamma = mne.time_frequency.psd_array_multitaper(channel, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
        new_data[f'C{c} gamma'] = np.mean(psd_gamma)
    return new_data


def getData():
    if not os.path.exists('data/data_preprocessed.csv'):
        dataframe = pd.DataFrame()
        data_repo = 'subjects_scans'
        subjects_repo = []
        for repo in os.listdir(data_repo):
            if os.path.isdir(data_repo + '/' + repo):
                subjects_repo.append(repo)

        breakpoint()
        for i, repo in enumerate(subjects_repo):
            printLog(f'\n =========   {i}/{len(subjects_repo)}   ========')
            subject_id = i + 1 
            files = os.listdir(data_repo + '/' + repo)
            for file in files:
                printLog(f"=======> Processing {file}...")
                file_path = data_repo + '/' + repo + '/' + file
                raw = mne.io.read_raw_edf(file_path, preload=True)
                annotations = raw.annotations

                for a, annotation in enumerate(annotations):
                    printLog(f'===========> annotation {a + 1}/{len(annotations)}')
                    start = int(annotation['onset'] * raw.info['sfreq'])
                    end = start + int(annotation['duration'] * raw.info['sfreq'])
                    segment = raw[:, start:end]
                    anno_type = annotation['description']
                    new_data = {}
                
                    new_data['id'] = subject_id
                    new_data['label'] = getLabel(file, anno_type)
                    new_data = getAlphaSignals(new_data, segment, raw.info['sfreq'])
                    new_data = getBetaSignals(new_data, segment, raw.info['sfreq'])
                    new_data = getGammaSignals(new_data, segment, raw.info['sfreq'])
                
                    tmp_df = pd.DataFrame([new_data])
                    dataframe = pd.concat([dataframe, tmp_df], ignore_index=True)
                printLog('=======> Done\n')
            dataframe.to_csv('data/data_preprocessed.csv', index=False)
    else:
        dataframe = pd.read_csv('data/data_preprocessed.csv')
    
    features = list(dataframe.columns)
    features.remove('label')
    features.remove('id')

    return dataframe, features


def UnderSample(X, y, random_state=42):
    np.random.seed(random_state)
    idx_to_remove = []
    labels, counts = np.unique(y, return_counts=True)
    min_count = min(counts)
    for label in labels:
        idx_list = y.index[y == label].tolist()
        while len(idx_list) > min_count:
            to_remove = random.choice(idx_list)
            idx_list.remove(to_remove)
            idx_to_remove.append(to_remove)
    X = X.drop(index=idx_to_remove)
    y = y.drop(index=idx_to_remove)
    return X, y

def OverSample(X, y, random_state=42):
    np.random.seed(random_state)
    labels, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    for label in labels:
        label_indexes = y.index[y == label].tolist()
        dup_nb = max_count - len(label_indexes)
        dup_indexes = np.random.choice(label_indexes, size=dup_nb, replace=True)
        X = pd.concat([X, X.loc[dup_indexes]], ignore_index=True)
        y = pd.concat([y, y.loc[dup_indexes]], ignore_index=True)
    return X, y