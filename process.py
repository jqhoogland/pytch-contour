import argparse, time, json

import numpy as np
import matplotlib.pyplot as plt

from speech.analysis.f0 import *
from speech.draw.pitch_contours import *
from speech.speech_file import *
from speech.speaker import *

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('-p', '--file-path', type=str, default="",
                    help='The location of the .wav audiofile to play.')

parser.add_argument('-P', '--pitch-range', type=int, nargs=2, default=[75, 500],
                    help='')

parser.add_argument('-O', '--octave-cost', type=float, default=0.01,
                    help='')

parser.add_argument('-V', '--voicing-threshold', type=float, default=0.45,
                    help='')

parser.add_argument('-S', '--silence-threshold', type=float, default=0.03,
                    help='')

parser.add_argument('-J', '--octave-jump-cost', type=float, default=0.35,
                    help='')

parser.add_argument('-T', '--voicing-transition-cost', type=float, default=0.14,
                    help='')

parser.add_argument('-c','--pitch-contour', type=str, default="boersma",
                    help='The kind of pitch_contour to apply. Options are currently restricted to `boersma``')

parser.add_argument('-a', '--audio', dest="audio", action="store_true",
                    help="Play the clip being analyzed. This is overriden by `-r`.")
parser.add_argument('--no_audio', dest="audio", action="store_false", help="Do not play the audio clip behind analyzed. Default.")
parser.set_defaults(audio=False)

parser.add_argument('-r','--record', dest="record", action="store_true",
                    help='Prompt the user to record their own voice. Overrides `-a`.')
parser.add_argument('--no-record', dest="record", action="store_false",
                    help='Prompt the user to not record their own voice. Default.')
parser.set_defaults(record=False)

parser.add_argument('-v','--verbose', dest="verbose", action="store_true",
                    help='Whether to record verbose output. Results in displaying more plots.')
parser.add_argument('--no-verbose', dest="verbose", action="store_false",
                    help='Hides additional plots. Default')
parser.set_defaults(verbose=False)


parser.add_argument('-s','--save', type=str, default="",
                    help='Path to save pitch contour to JSON. Default="" (does not save). ')

args = parser.parse_args()

if __name__ == "__main__":
    f0 = F0Drawer(F0Analyzer())

    settings = {
        "max_candidates_per_frame":15,
        "min_pitch":args.pitch_range[0],
        "max_pitch":args.pitch_range[1],
        "voicing_threshold":args.voicing_threshold,
        "silence_threshold":args.silence_threshold,
        "octave_cost":args.octave_cost,
        "voicing_transition_cost":args.voicing_transition_cost,
        "octave_jump_cost":args.octave_jump_cost
    }

    # plt.plot(f0.signal[0:500])
    # f0.listen()
    # f0.draw_spectrograms(args.frame_duration, args.frame_stride, args.n_fft_points, args.n_filters)

    res = {}
    src_path, target_path = np.array([]), np.array([])
    if args.file_path != "":
        speech_file = SpeechFile(args.file_path)

        if args.record:
            src_path, target_path = f0.record_and_compare(speech_file, **settings)
        else:
            _, src_path = f0.draw_f0_contour(speech_file, **settings)

            if args.audio:
                speech_file.play()


    else:
        _, target_path  = f0.record_and_draw(**settings)

    res["src_path"], res["target_path"] = src_path.tolist(), target_path.tolist()

    if args.save:
        print("Saving...")
        with open(args.save, "w+") as f:
            json.dump(res, f)

        print("Completed")

    plt.show()
