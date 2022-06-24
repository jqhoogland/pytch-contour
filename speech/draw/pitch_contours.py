"""
Boersma 1993 (Praat F0 Estimator)

"""
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from speech.analysis.f0 import *
from speech.speech_file import *


class F0Drawer:
    def __init__(self, f0_analyzer: F0Analyzer):
        self.f0_analyzer = f0_analyzer

    @staticmethod
    def align_paths(path_1, path_2):
        """Find the first instance of an entry greater than zero in both paths,
        and returns new versions of the paths aligned at this point. Also trims
        any trailing 0s at the end of the paths.

        At this point, it adds in 0s at the end to whichever path is shorter so
        that both are of equal length in the end.
        """
        first_idx_past_zero = lambda x: np.argmax(x > 0)
        last_idx_past_zero = lambda x: x.shape[0] - first_idx_past_zero(np.flip(x))

        path_1 = path_1[first_idx_past_zero(path_1) : last_idx_past_zero(path_1)]
        path_2 = path_2[first_idx_past_zero(path_2) : last_idx_past_zero(path_2)]

        # TODO: Perhaps there is a nicer, more compact way to do the next lines
        [shorter, longer] = np.sort([path_1.shape[0], path_2.shape[0]])
        zeros_to_append = np.zeros(longer - shorter)

        if path_1.shape[0] > path_2.shape[0]:
            path_2 = np.concatenate([path_2, zeros_to_append])
        elif path_2.shape[0] > path_1.shape[0]:
            path_1 = np.concatenate([path_1, zeros_to_append])

        return path_1, path_2

    def draw_f0_contour(self, speech_file, **kwargs):
        path = self.f0_analyzer.get_f0_contour(speech_file, **kwargs)
        first_idx_past_zero = np.argmax(path >= kwargs["min_pitch"])
        last_idx_past_zero = len(path) - np.argmax(path[::-1] >= kwargs["min_pitch"])

        trimmed_path = path[first_idx_past_zero:last_idx_past_zero]
        trimmed_path = path
        trimmed_path[trimmed_path == 0] = np.nan
        return plt.plot(trimmed_path), trimmed_path

    def animate_contour_analysis(self, speech_file, **kwargs):

        [
            path,
            speech_file,
            frames,
            windowed_frames,
            windowed_frames_autocorrelation,
            frames_autocorrelation,
            window_autocorrelation,
            min_pitch,
            max_pitch,
        ] = self.f0_analyzer.get_f0_contour(speech_file, full_output=True, **kwargs)

        n_frames, window_length = frames.shape

        fig, axes = plt.subplots(4, 3)
        (frames_plot,) = axes[0, 0].plot(frames[0, :])
        axes[0, 1].plot(self.f0_analyzer.get_window(window_length))
        (windowed_frames_plot,) = axes[0, 2].plot(windowed_frames[0, :])

        (windowed_autocorrelation_plot,) = axes[1, 0].plot(
            windowed_frames_autocorrelation[0, :]
        )
        axes[1, 1].plot(window_autocorrelation)
        (autocorrelation_plot,) = axes[1, 2].plot(
            frames_autocorrelation[0, : window_length // 2]
        )
        path_ax = plt.subplot(4, 3, (7, 9))
        path_ax.set_xlim(0, len(path))
        path_ax.set_ylim(min_pitch, max_pitch)
        (path_plot,) = path_ax.plot(path[:0])

        plt.subplot(4, 3, (10, 12)).plot(speech_file.signal)
        plt.show()

        def render(frame_idx):
            frames_plot.set_ydata(frames[frame_idx, :])
            autocorrelation_plot.set_ydata(
                frames_autocorrelation[frame_idx, : window_length // 2]
            )
            windowed_frames_plot.set_ydata(windowed_frames[frame_idx, :])
            windowed_autocorrelation_plot.set_ydata(
                windowed_frames_autocorrelation[frame_idx, :]
            )
            path_ax.plot(path[:frame_idx])
            return (
                frames_plot,
                autocorrelation_plot,
                windowed_frames_plot,
                windowed_autocorrelation_plot,
            )

        anim = animation.FuncAnimation(fig, render, frames=n_frames, interval=10)

    def record_and_draw(self, duration=2, sampling_rate=24000, file_path="", **kwargs):
        n_samples = int(sampling_rate * duration)

        print("Starting recording...")
        signal = sd.rec(n_samples, samplerate=sampling_rate, channels=1)
        speech_file = SpeechFile(
            file_path=file_path, sound_file=(signal, sampling_rate), overwrite=True
        )

        if file_path != "":
            speech_file.save()

        sd.wait()
        print("Recording done.")
        return self.draw_f0_contour(speech_file, **kwargs)

    def record_and_compare(
        self, src_speech_file, target_sampling_rate=24000, file_path="", **kwargs
    ):

        fig, (ax1, ax2) = plt.subplots(2, 1)

        src_path = self.f0_analyzer.get_f0_contour(src_speech_file, **kwargs)

        print("\n")

        src_speech_file.trim(src_path)
        src_speech_file.play()
        sd.wait()

        n_samples = (
            src_speech_file.signal.shape[0]
            * target_sampling_rate
            // src_speech_file.sampling_rate
        )

        print("Starting recording...")

        target_signal = sd.rec(n_samples, samplerate=target_sampling_rate, channels=1)

        target_speech_file = SpeechFile(
            file_path=file_path,
            sound_file=(target_signal, target_sampling_rate),
            overwrite=True,
        )

        if file_path != "":
            target_speech_file.save()

        sd.wait()

        print("Recording done.")

        target_path = self.f0_analyzer.get_f0_contour(target_speech_file, **kwargs)
        src_path, target_path = self.align_paths(src_path, target_path)

        ax1.set_title(src_speech_file.name)
        ax1.plot(src_path)
        ax2.plot(target_path)

        return src_path, target_path
