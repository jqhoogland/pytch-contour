"""
A class for describing human speech segments.
"""
import os

import numpy as np
import soundfile as sf
import sounddevice as sd

from speech.speaker import *
# TODO this should inherit from soundfile

class SpeechFile(object):
    def __init__(self,
                 file_path="",
                 sound_file=(),
                 language="English",
                 overwrite=False,
                 name=None,
                 speaker=None):

        self.file_path = file_path

        # For personal reference: when read as float64 (default), audio data is
        # typically between -1.0 and 1.0
        if not overwrite and os.path.isfile(file_path):
            self.signal, self.sampling_rate = sf.read(file_path)
        else: # Otherwise, use a provided sound_file= (signal, sampling_rate)
            self.signal, self.sampling_rate = sound_file

        self.language = language

        if name is None:
            # Remove parent directories, extension; "_" -> " "; Capitalize.
            name = os.path.basename(file_path)
            name = name[:name.rfind('.')].replace("_", " ").capitalize()
            print(name)

        self.name = name

        if speaker is None:
            speaker = Speaker(first_name="", age= -1, gender="o", languages=[language])

        self.speaker = speaker

    def save(self):
        sf.write(file_path, self.signal, self.sampling_rate)

    def play(self):
        sd.play(self.signal, self.sampling_rate)

    def trim(self, path, padding=0.2):
        n_samples_per_path_entry = self.signal.size / path.size

        first_idx_past_zero = lambda x: int(np.argmax(x > 0) * n_samples_per_path_entry * (1 - padding))
        last_idx_past_zero = lambda x: int((x.size * n_samples_per_path_entry - first_idx_past_zero(np.flip(x))) * (1 + padding))

        # path = path[first_idx_past_zero(path_1):last_idx_past_zero(path_1)]
        self.signal = self.signal[first_idx_past_zero(path):last_idx_past_zero(path)]

        return self.signal
