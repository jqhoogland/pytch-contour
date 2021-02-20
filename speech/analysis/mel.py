"""
Inspiration from: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html (MIT License)
ON mel-filter formula: http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/ (via above)

Estimating f0 contour: ROBUST F0 ESTIMATION IN NOISY SPEECH SPEECH_FILE.SIGNALS USING SHIFT AUTOCORRELATION
    (Frank Kurth, Alessia Cornaggia-Urrigshardt and Sebastian Urrigshardt)


Author: Jesse Hoogland
License: MIT
"""
import sys, argparse

import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

from speech.analysis.window import *

class MelAnalyzer(WindowAnalyzer):
    def compute_power_spectrum(self, frames, N=512):
        """
        Performs an N-point fast fourier transform (aka Short-Time FFT) to
        calculate the frequency_spectrum. Then calculate the power spectrum
        according to:

            P=|FFT(x_i)|^2/N,

        where, x_i is the ith frame of speech_file.signal x.

        """

        freq_spectrum = np.absolute(np.fft.rfft(frames, N))  # Magnitude of the FFT
        return ((1.0 / N) * ((freq_spectrum) ** 2))

    def apply_mel_filters(self, power_spectrum, N, n_filters=40):
        """
        Applies `n_filt`-many triangular filters to `power_spectrum` to extract
        frequency bands. This is the Mel-scale which human auditory non-linear
        perception; lower frequencies are better distinguishable than high
        frequencies.

        To convert between mel (m) and frequency or Hz (f):

            m = 2595 * log_(10)(1 + f * 700)
            f = 700 * (10 ^ (m / 2595) − 1)

        """
        low_freq_mel = 0

        convert_hz_to_mel = lambda hz: (2595 * np.log10(1 + hz / 700))
        convert_mel_to_hz = lambda mel: (700 * (10**(mel / 2595) - 1))

        # Derive intervals in hz-scale that are equally spaced on the mel-scale
        high_freq_mel = convert_hz_to_mel(self.speech_file.sampling_rate / 2)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
        hz_points = convert_mel_to_hz(mel_points)

        # converts the frequncies to fft bin numbers
        bin_idxs = np.floor((N + 1) * hz_points / self.speech_file.sampling_rate)

        # Create the appropriate set of triangular filters
        fbank = np.zeros((n_filters, int(np.floor(N / 2 + 1))))
        for m in range(1, n_filters + 1):
            f_m_minus = int(bin_idxs[m - 1])   # left
            f_m = int(bin_idxs[m])             # center
            f_m_plus = int(bin_idxs[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_idxs[m - 1]) / (bin_idxs[m] - bin_idxs[m - 1])

            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin_idxs[m + 1] - k) / (bin_idxs[m + 1] - bin_idxs[m])

        # Apply the mel-filters to the power-spectrum
        filter_banks = np.dot(power_spectrum, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        return filter_banks

    def get_spectrograms(self, speech_file, frame_duration=0.025, frame_stride=0.01, N=512, n_filters=40):
        frames = self.get_frames(speech_file.signal, speech_file.sampling_rate, frame_duration, frame_stride)
        frames = self.apply_window(frames)
        power_spectrum = self.compute_power_spectrum(frames, N)
        return self.apply_mel_filters(power_spectrum, N, n_filters)

    def draw_spectrograms(self, speech_file, frame_duration=0.025, frame_stride=0.01, N=512, n_filters=40):
        """
        Uses matplotlib's imshow to draw the FFTs performed over the audio file
        """
        spectrograms = self.get_spectrograms(speech_file.signal, speech_file.sampling_rate, frame_duration, frame_stride, N, n_filters)

        fig = plt.figure()
        ax = plt.axes()
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")

        # Draw the heatmap
        im = plt.imshow(spectrograms.T, cmap='jet', #extent=[ *x_extent, *y_extent],
                        origin='lowest', aspect='auto',
                        interpolation="none", animated=True)

        plt.colorbar(ax=ax) # must come after image is drawn
        plt.show()
