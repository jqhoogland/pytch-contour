import argparse, time, json
from numbers import Number, Real
from typing import TypedDict
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import plotext as plx
import yaml

from speech.analysis.f0 import *
from speech.draw.pitch_contours import *
from speech.speech_file import *
from speech.speaker import *

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "-p",
    "--file-path",
    type=str,
    default="",
    help="The location of the .wav audiofile to play.",
)

parser.add_argument(
    "-P",
    "--pitch-range",
    type=int,
    nargs=2,
    default=[75, 500],
    help="Default: [75, 500]",
)

parser.add_argument(
    "-O", "--octave-cost", type=float, default=0.01, help="Default: 0.01"
)

parser.add_argument(
    "-V", "--voicing-threshold", type=float, default=0.45, help="Default: 0.45"
)

parser.add_argument(
    "-S", "--silence-threshold", type=float, default=0.03, help="Default: 0.03"
)

parser.add_argument(
    "-J", "--octave-jump-cost", type=float, default=0.35, help="Default: 0.35"
)

parser.add_argument(
    "-T", "--voicing-transition-cost", type=float, default=0.14, help="Default: 0.14"
)

parser.add_argument(
    "-c",
    "--pitch-contour",
    type=str,
    default="boersma",
    help="The kind of pitch_contour to apply. Options are currently restricted to `boersma` (also the default)",
)

parser.add_argument(
    "-a",
    "--audio",
    dest="audio",
    action="store_true",
    help="Play the clip being analyzed. This is overriden by `-r`.",
)
parser.add_argument(
    "--no_audio",
    dest="audio",
    action="store_false",
    help="Do not play the audio clip behind analyzed. This is the default option.",
)
parser.set_defaults(audio=False)

parser.add_argument(
    "-r",
    "--record",
    dest="record",
    action="store_true",
    help="Prompt the user to record their own voice. Overrides `-a`.",
)
parser.add_argument(
    "--no-record",
    dest="record",
    action="store_false",
    help="Prompt the user to not record their own voice. This is the default option.",
)
parser.set_defaults(record=False)

parser.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action="store_true",
    help="Whether to record verbose output. Results in displaying more plots.",
)
parser.add_argument(
    "--no-verbose",
    dest="verbose",
    action="store_false",
    help="Hides additional plots. This is the default option.",
)
parser.set_defaults(verbose=False)

parser.add_argument(
    "-s",
    "--save",
    type=str,
    default="",
    help="Path to save pitch contour to (JSON). The default is to not save. ",
)

parser.add_argument(
    "-d", "--duration", type=float, default=2, help="Duration (in seconds)"
)

parser.add_argument(
    "-C", "--config", type=str, default="", help="Path to config file (.yaml)"
)


class ContourSettings(TypedDict):
    max_candidates_per_frame: int
    min_pitch: Real
    max_pitch: Real
    voicing_threshold: Real
    silence_threshold: Real
    octave_cost: Real
    voicing_transition_cost: Real
    octave_jump_cost: Real
    duration: float  # seconds


@dataclass
class PytchContoursOptions:
    contours: ContourSettings
    file_path: str | None
    audio: bool
    record: bool
    save: str | None


def run_analysis(options: PytchContoursOptions):
    f0 = F0Drawer(F0Analyzer())

    res = {}
    src_path, target_path = np.array([]), np.array([])
    has_reference_file = options.file_path != ""

    if has_reference_file:
        speech_file = SpeechFile(options.file_path)

        if options.record:
            src_path, target_path = f0.record_and_compare(
                speech_file, **options.contours
            )
        else:
            _, src_path = f0.draw_f0_contour(speech_file, **options.contours)

            if options.audio:
                speech_file.play()
    else:
        _, target_path = f0.record_and_draw(**options.contours)

    res["src_path"], res["target_path"] = src_path.tolist(), target_path.tolist()

    if options.save:
        print("Saving...")
        with open(options.save, "w+") as f:
            json.dump(res, f)

        print("Saved")


if __name__ == "__main__":
    args = parser.parse_args()

    contour_settings = {
        "max_candidates_per_frame": 15,
        "min_pitch": args.pitch_range[0],
        "max_pitch": args.pitch_range[1],
        "voicing_threshold": args.voicing_threshold,
        "silence_threshold": args.silence_threshold,
        "octave_cost": args.octave_cost,
        "voicing_transition_cost": args.voicing_transition_cost,
        "octave_jump_cost": args.octave_jump_cost,
        "duration": args.duration,
    }

    if args.config:
        with open(args.config) as f:
            contour_settings |= yaml.safe_load(f) or {}

    print("\n")
    print("-" * 30 + " Pytch Contours " + "-" * 30)
    print("\n")

    settings_str = "  " + yaml.dump(contour_settings).replace("\n", "\n  ")
    print(f"Contour Settings:\n{settings_str}")

    options = PytchContoursOptions(
        contours=contour_settings,
        file_path=args.file_path,
        record=args.record,
        save=args.save,
        audio=args.audio,
    )

    i = 0

    while True:
        run_analysis(options)

        plx.from_matplotlib(plt.gcf())
        plx.show()

        action = input("Type Q to exit. Type R to reset").lower()

        if action == "q":
            break
        elif action == "r":
            plt.gcf().clear(True)

        i += 1
