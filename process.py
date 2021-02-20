import argparse, time, json

import numpy as np
import matplotlib.pyplot as plt

from speech.analysis.f0 import *
from speech.draw.pitch_contours import *
from speech.speech_file import *
from speech.speaker import *

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('-p', '--file_path', type=str, default="",
                    help='The location of the .wav audiofile to play.')

parser.add_argument('-P', '--pitch_range', type=int, nargs=2, default=[75, 500],
                    help='')

parser.add_argument('-O', '--octave_cost', type=float, default=0.01,
                    help='')

parser.add_argument('-V', '--voicing_threshold', type=float, default=0.45,
                    help='')

parser.add_argument('-S', '--silence_threshold', type=float, default=0.03,
                    help='')

parser.add_argument('-J', '--octave_jump_cost', type=float, default=0.35,
                    help='')

parser.add_argument('-T', '--voicing_transition_cost', type=float, default=0.14,
                    help='')

parser.add_argument('-c','--pitch_contour', type=str, default="boersma",
                    help='The kind of pitch_contour to apply. Options are currently restricted to `boersma``')

parser.add_argument('-r','--record', type=str, default="False",
                    help='Whether to record input')

parser.add_argument('-v','--verbose', type=str, default="False",
                    help='Whether to record verbose output. Results in displaying more plots.')

parser.add_argument('-s','--save', type=str, default="",
                    help='Path to save pitch contour to JSON. Default="" (does not save). ')

args = parser.parse_args()

if __name__ == "__main__":
    praat = F0Drawer(F0Analyzer())
    # time.sleep(1)

    settings = {"max_candidates_per_frame":15,
                "min_pitch":args.pitch_range[0],
                "max_pitch":args.pitch_range[1],
                "voicing_threshold":args.voicing_threshold,
                "silence_threshold":args.silence_threshold,
                "octave_cost":args.octave_cost,
                "voicing_transition_cost":args.voicing_transition_cost,
                "octave_jump_cost":args.octave_jump_cost}

    # plt.plot(praat.signal[0:500])
    # praat.listen()
    # praat.draw_spectrograms(args.frame_duration, args.frame_stride, args.n_fft_points, args.n_filters)

    res = {}
    src_path, target_path = np.array([]), np.array([])
    if args.file_path != "":
        speech_file = SpeechFile(args.file_path)

        if args.record == "True":
            src_path, target_path = praat.record_and_compare(speech_file, **settings)
        else:
            _, src_path = praat.draw_f0_contour(speech_file, **settings)

    else:
        _, target_path  = praat.record_and_draw(**settings)

    res["src_path"], res["target_path"] = src_path.tolist(), target_path.tolist()

    if args.save:
        print("Saving...")
        with open(args.save, "w+") as f:
            json.dump(res, f)

        print("Completed")

    plt.show()
