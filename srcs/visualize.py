import mne
import sys
import argparse
import matplotlib.pyplot as plt

from colorama import Fore, Style
from utils.tools import printLog, printError

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('edf_file', help='The edf file')
    return parser.parse_args()

def AlphaSpectre(segment, sfreq):
    fmin, fmax = 8, 13
    for c, channel in enumerate(segment[0]):
        psd, freqs = mne.time_frequency.psd_array_multitaper(channel, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
        plt.plot(freqs, psd)
        plt.title(f'Alpha spectre - Channel {c + 1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB/Hz)')
        plt.show()


def BetaSpectre(segment, sfreq):
    fmin, fmax = 13, 30
    for c, channel in enumerate(segment[0]):
        psd, freqs = mne.time_frequency.psd_array_multitaper(channel, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
        plt.plot(freqs, psd)
        plt.title(f'Beta spectre - Channel {c + 1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB/Hz)')
        plt.show()

def GammaSpectre(segment, sfreq):
    fmin, fmax = 30, 45
    for c, channel in enumerate(segment[0]):
        psd, freqs = mne.time_frequency.psd_array_multitaper(channel, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
        plt.plot(freqs, psd)
        plt.title(f'Gamma spectre - Channel {c + 1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB/Hz)')
        plt.show()


if __name__ == '__main__':
    try:
        args = parse()
        raw_edf = mne.io.read_raw_edf(args.edf_file, preload=True)
        graph_type = input(Fore.GREEN + f'\nVisualisation options:\n 1 ==> Raw data\n 2 ==> Spectral forms\n\n===> ' + Style.RESET_ALL).strip()

        if graph_type == '1': 
            raw_edf.plot()
            plt.show()
        
        elif graph_type == '2':
            segment = raw_edf[:, :]
            signal_type = input(Fore.GREEN + f'Signal type:\n 1 ==> Alpha\n 2 ==> Beta\n 3 ==> Gamma\n\n===> ' + Style.RESET_ALL).strip()
            if signal_type == '1':
                AlphaSpectre(segment, raw_edf.info['sfreq'])
            elif signal_type == '2':
                BetaSpectre(segment, raw_edf.info['sfreq'])
            elif signal_type == '3':
                GammaSpectre(segment, raw_edf.info['sfreq'])
            else:
                printError(f'Error: input not available')
        else:
            printError(f'Error: input not available')


    except Exception as error:
        print(error)