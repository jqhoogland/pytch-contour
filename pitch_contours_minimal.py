"""
Inspiration from: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html (MIT License)

Author: Jesse Hoogland
License: MIT
"""
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

from utils import debug_print

np.set_printoptions(threshold=sys.maxsize)

DEBUG = True

class PitchDrawer(object):
    def __init__(self, file_path, force_mono=False):
        self.load(file_path, force_mono)
        self.listen()

    def load(self, file_path, force_mono=False):
        """ Loads the audio file stored at file_path.
            Defines self.file_path, self.signal (np.array), self.samping_rate
        """
        self.file_path = file_path
        self.signal, self.sampling_rate = sf.read(self.file_path, dtype='float32')
        self.n_samples, self.n_channels= self.signal.shape

        # DEBUG
        force_mono = True

        if force_mono and self.n_channels > 1:
            # For convenience we temporarily ignore the second channel
            self.signal = self.signal[:, 0].reshape((-1, 1))
            self.n_channels = 1

    def listen(self):
        """
        Loads the audio from self.filename.
        Provides this signal to the processing functionalityas a real-time stream.

        Called `listen` rather than `load` to reflect functionality in
            RadarListener
        """
        sd.play(self.signal, self.sampling_rate)

    def get_spectrograms(self, n_samples_per_bucket=100, gain=100):
        """ Returns real fast fourier transform performed on the audio signal in
        self.signal, in buckets with `n_samples_per_bucket` samples.

        TODO: Use mel-frequency cepstrum spectrograms (captures human
        voice details better)
        """
        n_buckets = self.n_samples // n_samples_per_bucket

        # cut off the problem_bit
        self.signal = self.signal[:(n_buckets * n_samples_per_bucket), :]
        _signal = self.signal.reshape([n_buckets, n_samples_per_bucket, self.n_channels])
        return np.abs(np.fft.rfft(_signal, axis=1)) * gain

    def get_fundamental_frequencies(self, n_samples_per_bucket=100, n_fundamentals=3):
        """
        Returns locally dominant frequencies.
        """
        spectrograms = self.get_spectrograms(n_samples_per_bucket)

        rolled_forward = np.roll(spectrograms, 1, axis=1)
        rolled_back = np.roll(spectrograms, -1, axis=1)

        # Fix the edge-cases
        rolled_forward[:, 0, :] = rolled_forward[:, 1, :]
        rolled_back[:, -1, :] = rolled_back[:, -2, :]

        threshold = 50

        are_peaks = np.logical_and(spectrograms - rolled_forward >= threshold, spectrograms - rolled_back >= threshold)

        # debug_print(DEBUG, "Entry [100,:,0]\nSpectrograms:\n{}\n\nRolled:\n{}\n\nCompared:\n{}\n\nPeaks:\n{}\n\n".format(
        #     spectrograms[100, :, 0],
        #     rolled_forward[100, :, 0],
        #     (spectrograms >= rolled_forward)[100, :, 0],
        #     are_peaks[100, :, 0]))

        peak_rows, peak_cols, peak_channels = np.where(are_peaks == 1)

        fundamental_frequencies = np.zeros((spectrograms.shape[2], spectrograms.shape[0], n_fundamentals))
        for i, (row_idx, col_idx, channel_idx) in enumerate(zip(peak_rows, peak_cols, peak_channels)):
            for j in fundamental_frequencies[channel_idx, row_idx, :]:
                j = int(j)
                if j == 0:
                    fundamental_frequencies[channel_idx, row_idx, j] = col_idx

        #debug_print(DEBUG, fundamental_frequencies)
        return fundamental_frequencies

    def get_f0s(self, n_samples_per_bucket=100):
        return self.get_fundamental_frequencies(n_samples_per_bucket, n_fundamentals=1).reshape((-1, ))

    def draw_spectrograms(self, n_samples_per_bucket=100):
        """
        Uses matplotlib's imshow to draw the FFTs performed over the audio file
        """
        spectrograms = self.get_spectrograms(n_samples_per_bucket)

        fig = plt.figure()
        ax = plt.axes()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Time (s)")

        # Draw the heatmap
        im = plt.imshow(spectrograms[:, :, 0], cmap='jet', #extent=[ *x_extent, *y_extent],
                        vmin= 0, vmax= 100, origin='lowest', aspect='auto',
                        interpolation="none", animated=True)

        plt.colorbar(ax=ax) # must come after image is drawn
        plt.show()

    def draw_f0s(self, n_samples_per_bucket=100):
        """
        Uses matplotlib's imshow to draw the FFTs performed over the audio file
        """
        f0s = self.get_f0s(n_samples_per_bucket)
        plt.plot(f0s)
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.show()



if __name__ == "__main__":
    pitch_drawer = PitchDrawer("tones/pian2yi0.wav", force_mono=True)
    #pitch_drawer.draw_spectrograms(1000)
    pitch_drawer.draw_f0s(500)
