"""OA
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

class WindowAnalyzer(object):
    def __init__(self, window="gaussian"):
        self.window_name=window

    def get_frames(self, speech_file, frame_duration=0.025, frame_stride=0.01):
        """
        Splits data into short-time frames (with some possible overlap).
        Frequencies will vary across our time intervals, and our frames are
        chosen such that we can assume the frequencies are constant within a
        frame.

        Then, we apply a Hamming window to thesen frames, which is the following
        function:

            w[n]=0.54−0.46*cos((2πn)/(N−1)),

        where 0≤n≤N−1, N is the window length.

        This serves to counteract the FFT's assumption of infinite data
        and reduces spectral leakage.

        Args:
            `frame_duration` (float): the duration of a frame (in seconds)
            `frame_stride` (float): the stride (in seconds)
                (i.e. frame_overlap = frame_duration-frame_stride)
        """

        # Convert from seconds to samples
        frame_length= round(frame_duration * speech_file.sampling_rate)
        frame_step = round(frame_stride * speech_file.sampling_rate)
        speech_file.signal_length = speech_file.signal.shape[0]

        # Ensure we have at least 1 frame
        # TODO: replace with the requirement that our audio fragment is longer
        # than the frame_duration
        num_frames = int(np.ceil(np.abs(speech_file.signal_length - frame_length) / frame_step))

        # Pad Speech_File.Signal to make sure that all frames have equal number of samples
        # without truncating any samples from the original speech_file.signal
        padding_length = num_frames * frame_step + frame_length
        padded_signal = np.append(speech_file.signal, np.zeros((padding_length - speech_file.signal_length)))

        indices = (np.tile(np.arange(0, frame_length),
                           (num_frames, 1))
                   + np.tile(np.arange(0, num_frames * frame_step, frame_step),
                             (frame_length, 1)).T)

        frames = padded_signal[indices.astype(np.int32, copy=False)]
        return frames

    def get_window(self, window_length):
        if self.window_name == "hamming":
            window = np.hamming(window_length)
        elif self.window_name == "hanning":
            window = np.hanning(window_length)
        elif self.window_name == "gaussian":
            # (exp(−12(t/T − 1/2)^2 ) − e^(−12))/(1− e^(−12) )
            ts = np.linspace(0, window_length - 1, window_length)
            window = (np.exp(-12 * np.square(ts / window_length - 0.5)) - np.exp(-12)) / (1 - np.exp(-12))
            print(window.shape, window_length)
        else:
            raise ValueError("Window {} is inappropriate choice. Must be one of [\"hanning\", \"hamming]".format(window))

        return window

    def apply_window(self, frames):
        window = [self.get_window(frames.shape[-1])]
        return frames * window
