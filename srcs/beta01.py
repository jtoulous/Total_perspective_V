import sys
import mne
import matplotlib.pyplot as plt
import numpy as np

from utils.tools import printError, printLog

if __name__ == "__main__":
    raw = mne.io.read_raw_edf(sys.argv[1], preload=True)
    breakpoint()
    raw.plot()
    plt.show()

#    data, _ = raw[:, :]
#
#    sfreq = raw.info['sfreq']

    #for dt in data:
    #    freqs, psd = plt.psd(dt, Fs=sfreq, NFFT=1024, pad_to=1024, noverlap=0)
    #    plt.title('PSD du signal')
    #    plt.xlabel('Fréquence (Hz)')
    #    plt.ylabel('Puissance spectrale (dB/Hz)')
    #    plt.show()