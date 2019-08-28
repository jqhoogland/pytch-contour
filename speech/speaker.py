"""
A class for describing human speech segments.
"""

import soundfile as sf

LANGUAGES = ["English", "Mandarin"]
LEVELS = ["Native", "A1", "A2", "B1", "B2", "C1", "C2"]
GENDERS = ["m", "f", "o"]

class Speaker(object):
    def __init__(self, first_name, age, gender, languages):
        """

        Args:
            `first_name` (str): The first name of the speaker
            `age` (int): The age of the speaker (in years)
            `gender` (char of 'm', 'f', 'o'): 'm', Male; 'f', Female; 'o', Other.
            `languages` (List of str or List of 2-Tuples of str):
                e.g. ["English", "Mandarin"] is equivalent to
                     [("English", "Native"), ("Mandarin", "Native)]
                Language_level must be one of:
                    ["Native", "A1", "A2", "B1", "B2", "C1", "C2"]
        """

        self.first_name = first_name
        self.age = age

        if gender not in GENDERS:
            raise ValueError("Inappropriate genderselection: {}. Gender must be one of {}".format(gender), GENDERS)

        self.gender = gender

        # VERIFY APPROPRIATE CHOICES OF LANGUAGE

        def verify_language_tuple(language, level):
            if language not in LANGUAGES:
                raise ValueError("Inappropriate language selection: {}. Language must be one of {}".format(language), LANGUAGES)
            if level not in LEVELS:
                raise ValueError("Inappropriate level selection: {}. Language must be one of {}".format(level), Levels)
            return language, level

        _languages = []
        for language in languages:
            if type(language) == str:
                _languages.append((language, "Native"))
            else:
                _languages.append(verify_language_tuple(*language))

        self.languages = _languages

        # CALIBRATE
        self.calibrate()

    def calibrate(self, min_pitch=50, max_pitch=500):
        """ Derives an appropriate min_ and max_pitch for the speaker by running
        a few tests.

        Args:
            min_pitch (int): Optional starting point for the minimum pitch (Hze
               (Default: 50))
            max_pitch (int): Optional starting point for the maximum pitch (Hze
               (Default: 50))
        """

        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

        return self.min_pitch, self.max_pitch
