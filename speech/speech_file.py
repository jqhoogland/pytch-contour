"""
A class for describing human speech segments.
"""
import os

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

        if not overwrite and os.path.isfile(file_path):
            self.signal, self.sampling_rate = sf.read(file_path)
        else: # Otherwise, use a provided sound_file= (signal, sampling_rate)
            self.signal, self.sampling_rate = sound_file

        self.language = language

        if name is None:
            # Remove extension; "_" -> " "; Capitalize.
            name = file_path[:file_path.rfind('.')].replace("_", " ").capitalize()

        self.name = name

        if speaker is None:
            speaker = Speaker(first_name="", age= -1, gender="o", languages=[language])

        self.speaker = speaker

    def save(self):
        sf.write(file_path, self.signal, self.sampling_rate)

    def play(self):
        sd.play(self.signal, self.sampling_rate)
