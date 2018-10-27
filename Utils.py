import scipy.signal
import soundfile as sf
import numpy as np
import os

def read_wav(filename):
    # Reads in a wav audio file, averages both if stereo, converts the signal to float64 representation

    audio_signal, sample_rate = sf.read(filename)

    if audio_signal.ndim > 1:
        audio_signal = (audio_signal[:, 0] + audio_signal[:, 1])/2.0

    if audio_signal.dtype != 'float64':
        audio_signal = wav_to_float(audio_signal)

    return audio_signal, sample_rate

def load_wav(wav_path, desired_sample_rate=22050):

    sequence, sample_rate = read_wav(wav_path)
    sequence = ensure_sample_rate(sequence, desired_sample_rate, sample_rate)
    return sequence

def ensure_sample_rate(x, desired_sample_rate, file_sample_rate):

    if file_sample_rate != desired_sample_rate:
        return scipy.signal.resample_poly(x, desired_sample_rate, file_sample_rate)
    return x

def wav_to_float(x):

    try:
        max_value = np.iinfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    except:
        max_value = np.finfo(x.dtype).max
        min_value = np.finfo(x.dtype).min
    x = x.astype('float64', casting='safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    return x

def load_tracks(dataset_path, tracks):
    """
    :param tracks: list of tracks, [music 1, music 2, ... ]

    :return: list of tuples (accompany, voice), [ (accompany, voice), (accompany, voice) ... ]
    """

    sequences = []
    for track in tracks:
        track = os.path.join(dataset_path, track)

        accompany_path = track + '.stem_accompaniment.wav'
        vocal_path = track + '.stem_vocals.wav'

        accompany = load_wav(accompany_path)
        vocal = load_wav(vocal_path)

        sequences.append((accompany, vocal))
    return sequences