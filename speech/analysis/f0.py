"""
Based on Boersma 1993 (Praat F0 Estimator)

"""
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from speech.analysis.window import *
from speech.speech_file import *

np.set_printoptions(threshold=1000)
# TODO Configure correct constant scaling factor
lag_freq_switch= lambda lag_or_freq, sampling_rate: sampling_rate / lag_or_freq
lag_freq_switch_int = lambda lag_or_freq, sampling_rate: sampling_rate // lag_or_freq

def sin_x_by_x_interpolation(tau_prime, samples):
    _, window_length = samples.shape
    # TODO: make this work with array-valued tau
    # Continuous sin x / x fit to the samples `frames_autocorrelation`
    n_l = np.floor(tau_prime).astype(np.int32)
    n_r = n_l + 1
    phi_l = tau_prime - n_l
    phi_r = 1 - phi_l
    N = window_length // 2 #min(500, window_length // 2)
    sin_term = lambda n, phi: ((np.sin(np.pi * (phi + n - 1))) / (np.pi * (phi + n - 1)))
    cos_term = lambda n, phi: (0.5 + 0.5 * np.cos((np.pi * (phi + n - 1)) / (phi + N)))
    ns = np.arange(1, N + 1)
    result = (samples[n_r - ns] * sin_term(ns, phi_l) * cos_term(ns, phi_l)
              + samples[n_l + ns] * sin_term(ns, phi_r) * cos_term(ns, phi_r))
    # Only consider values between min_pitch and max_pitch
    # DO something

    return np.sum(result, axis= -1)

def parabolic_interpolation(correlation, lag_idx, min_pitch, sampling_rate, octave_cost):
    d_correlation = 0.5 * (correlation[lag_idx + 1]
                           - correlation[lag_idx - 1])
    d2_correlation = (2 * correlation[lag_idx]
                      - correlation[lag_idx - 1]
                      - correlation[lag_idx + 1])
    max_lag = lag_idx + d_correlation / d2_correlation
    max_freq = lag_freq_switch(max_lag, sampling_rate)
    max_strength = correlation[lag_idx] + (2 * d_correlation) ** 2 / (8 * d2_correlation)
    max_strength = max_strength - octave_cost * np.log(min_pitch * max_lag)

    if (max_strength > 1):
        max_strength = 1 / max_strength


    return max_freq, max_strength

class F0Analyzer(WindowAnalyzer):
    def preprocess(self, speech_file, N=512):
        """ STEP 1: remove the sidelobe of the Fourier transform of the
        Hanning window for signal components near the Nyquist frequency, we perform a
        soft upsampling as follows: do an FFT on the whole signal; filter by multiplication in
        the frequency domain linearly to zero from 95% of the Nyquist frequency to 100% of
        the Nyquist frequency; do an inverse FFT of order one higher than the first FFT.
        """
        nyquist_freq = speech_file.sampling_rate / 2
        lower_limit, upper_limit = nyquist_freq * 0.95, nyquist_freq
        fft_spectrum = np.fft.rfft(signal, N)
        fft_freqs = np.fft.fftfreq(N)
        fft_spectrum[np.where(np.logical_and((fft_freqs <= upper_limit), (fft_freqs >= lower_limit)))] = 0
        processed_signal = np.fft.ifft(fft_spectrum, N + 1)
        return processed_signal

    def get_global_max(self, signal):
        """ Step 2. Compute the global absolute max value of the signal (see step 3.3).
        """
        return np.max(signal)

    def get_window_length(self, sampling_rate, min_pitch=75):
        """
        The window should be just long
        enough to contain three periods (for pitch detection) or six periods (for HNR
        measurements) of MinimumPitch. E.g. if MinimumPitch is 75 Hz, the window length
        is 40 ms for pitch detection and 80 ms for HNR measurements

        Args:
            `min_pitch` (int): the minimum allowed frequency (Hz) allowed for f0.

        Returns:
            window_length (int): the duration of the window in terms of the
                number of samples
            window_duration (float): the duration of the window in terms of number of seconds
        """
        window_duration = (3 / min_pitch)
        return  int(window_duration * sampling_rate), window_duration

    def get_autocorrelation(self, frames):
        """
        frames (np.array): Of either shape (num_frames, window_length) or (window_length)
        """

        window_length = frames.shape[-1]
        # Step 3.5. Append half a window length of zeroes (because we need autocorrelation
        # values up to half a window length for interpolation).
        # Step 3.6. Append zeroes until the number of samples is a power of two.

        num_zeros_to_add = int(2 ** np.ceil(np.log2(3 * window_length / 2)) - window_length)
        zeros_shape = [ *frames.shape]
        zeros_shape[-1] = num_zeros_to_add

        frames = np.append(frames,
                           np.zeros(zeros_shape),
                           axis= -1)

        # Step 3.7. Perform a Fast Fourier Transform (discrete version of equation 15)
        # TODO: check whether step 3.6 is really necessary (np's fft can automatically append zeros)
        freq_spectrum = np.fft.fft(frames, axis=-1)
        # Step 3.8. Square the samples in the frequency domain
        power_spectrum = np.square(np.absolute(freq_spectrum))
        # Step 3.9. Perform an inverse Fast Fourier Transform (discrete version of equation 16). This
        # gives a sampled version of ra(τ).
        lag_spectrum = np.abs(np.fft.ifft(power_spectrum, axis= -1))

        return (lag_spectrum / np.max(lag_spectrum, axis= -1, keepdims=True))

    def get_best_path(self,
                      freq_strength_pairs,
                      voicing_transition_cost,
                      octave_jump_cost):
        """
        """

        [n_frames, max_candidates_per_frame, _]= freq_strength_pairs.shape

        def transition_cost(f_before, f_after):
            if f_before == 0 and f_after == 0:
                return 0
            elif f_before == 0 or f_after == 0: # Effectively an XOR
                return voicing_transition_cost
            else:
                #return octave_jump_cost ** 2 * np.absolute(np.log (f_before / f_after))
                return octave_jump_cost * np.absolute(np.log2 (f_before / f_after))

        path = np.zeros(n_frames)
        possibility_costs = np.zeros(max_candidates_per_frame)

        path[0] = freq_strength_pairs[0, freq_strength_pairs[0, :, 0].argsort()[0], 0]
        for frame_idx in range(1, n_frames):
            prev_frame = freq_strength_pairs[frame_idx - 1]
            curr_frame = freq_strength_pairs[frame_idx]
            prev_choice = path[frame_idx - 1]
            for candidate_idx in range(max_candidates_per_frame):
                [candidate_choice_freq, candidate_choice_strength] = curr_frame[candidate_idx]
                possibility_costs[candidate_idx] = transition_cost(prev_choice, candidate_choice_freq) - candidate_choice_strength

            path[frame_idx] = curr_frame[(possibility_costs).argsort()][0, 0]

        return path

    def get_f0_contour(self,
                       speech_file,
                       max_candidates_per_frame=4,
                       min_pitch=100,
                       max_pitch=900,
                       voicing_threshold=0.4,
                       silence_threshold=0.05,
                       octave_cost=0.2,
                       voicing_transition_cost=0.2,
                       octave_jump_cost=0.2,
                       full_output=False):

        assert max_pitch > min_pitch and max_pitch < speech_file.sampling_rate / 2

        # TODO: Test preprocessing
        # signal = self.preprocess(signal, sampling_rate)

        global_max = self.get_global_max(speech_file.signal)
        window_length, window_duration= self.get_window_length(speech_file.sampling_rate, min_pitch)
        frame_stride = 0.75 / min_pitch

        # Steps 3.1-3.4
        #
        #    a(t) = x * (t_mid − 1/2 * T + t) − µ * w(t),
        #
        # where T is the duration of the frame, t_mid is the midpoint time of the
        # frame, µ is the mean of the frame, and w(t) is the window function

        # Step 3.1

        frames = self.get_frames(speech_file, frame_duration=window_duration, frame_stride=frame_stride)
        [n_frames, _] = frames.shape

        # Step 3.3 Determine local maxima (to check if unvoiced later)
        local_maxes = np.max(frames, axis=1, keepdims=True)

        # Step 3.2 Subtract the local average
        frames -= np.mean(frames, axis=1, keepdims=True) #TODO check broadcasting

        # Step 3.4 Multiply by the window function
        windowed_frames = self.apply_window(frames)

        # # Steps 3.5-3.9 Determine the autocorrelation of the windowed frames
        windowed_frames_autocorrelation = self.get_autocorrelation(windowed_frames)[:, :window_length]

        # Step 3.10. Divide by the autocorrelation of the window, which was computed once
        # with steps 3.5 through 3.9 (equation 9). This gives a sampled version of rx(τ).;
        window_autocorrelation = self.get_autocorrelation(self.get_window(window_length))[:window_length]
        frames_autocorrelation = (windowed_frames_autocorrelation / window_autocorrelation)


        # Step 3.11. Find the places and heights of the maxima of the continuous version of
        # rx(τ), which is given by equation 22, e.g., with the algorithm brent from Press et al.
        # (1989). The only places considered for the maxima are those that yield a pitch
        # between MinimumPitch and MaximumPitch.

        # TODO Check working of sinx by x interpolation (currently using parabolic interpolation)

        maxima = np.zeros((n_frames, max_candidates_per_frame - 1, 2)) # last index is a pair (time, strength)

        # TODO Configure correct limits here
        min_lag_idx = min_pitch #lag_freq_switch_int(max_pitch, speech_file.sampling_rate)
        max_lag_idx = max_pitch #lag_freq_switch_int(min_pitch, speech_file.sampling_rate)

        for frame_idx in range(n_frames):
            # frame_maxima is a list of length-two arrays (maximum_frequency, corresponding_strength)
            frame_maxima = []
            for lag_idx in range(min_lag_idx, min(max_lag_idx - 1, window_length - 1)):
                frame = frames_autocorrelation[frame_idx]

                is_local_max = (frame[lag_idx] >= frame[lag_idx - 1] and frame[lag_idx] >= frame[lag_idx + 1])
                exceeds_voicing_threshold = frame[lag_idx] > 0.5 * voicing_threshold

                if (is_local_max and exceeds_voicing_threshold):
                    frame_maxima.append(parabolic_interpolation(frame, lag_idx, min_pitch, speech_file.sampling_rate, octave_cost))

            # Reshape into the correct format, sort according to cost
            frame_maxima = np.asarray(frame_maxima).reshape((-1, 2))
            frame_maxima = (frame_maxima[-frame_maxima[:, 1].argsort()])
            frame_maxima = frame_maxima[:max_candidates_per_frame - 1]

            maxima[frame_idx, : len(frame_maxima)] = frame_maxima

        max_arg = 2 - (local_maxes / global_max)* ((1 + voicing_threshold) / silence_threshold)
        max_arg = np.concatenate([np.zeros(max_arg.shape), max_arg], axis=1)
        unvoiced_strengths = (voicing_threshold
                              + np.max(max_arg, axis=1)).reshape((n_frames, 1))

        unvoiced_pairs = np.concatenate((np.zeros((n_frames, 1)), unvoiced_strengths), axis=1)
        unvoiced_pairs = unvoiced_pairs.reshape((n_frames, 1, 2))
        maxima = np.concatenate((unvoiced_pairs, maxima), axis= 1)


        # After performing step 2 for every frame, we are left with a number of frequencystrength pairs (Fni, Rni), where the index n runs from 1 to the number of frames, and i
        # is between 1 and the number of candidates in each frame. The locally best candidate
        # in each frame is the one with the highest R. But as we can have several approximately
        # equally strong candidates in any frame, we can launch on these pairs the global path
        # finder, the aim of which is to minimize the number of incidental voiced-unvoiced
        # decisions and large frequency jumps:

        path = self.get_best_path(maxima, voicing_transition_cost, octave_jump_cost)

        if full_output:
            res = [path,
                   speech_file.signal,
                   frames,
                   windowed_frames,
                   windowed_frames_autocorrelation,
                   frames_autocorrelation,
                   window_autocorrelation,
                   min_pitch,
                   max_pitch]
        else:
            res = path

        return res
