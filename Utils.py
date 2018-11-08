import scipy.signal
import soundfile as sf
import numpy as np
import os

import warnings


def read_wav(filename, stereo=True):
    # Reads in a wav audio file, averages both if stereo, converts the signal to float64 representation

    audio_signal, sample_rate = sf.read(filename)

    if audio_signal.ndim > 1 and not stereo:
        audio_signal = (audio_signal[:, 0] + audio_signal[:, 1])/2.0
        audio_signal = np.expand_dims(audio_signal, 1)

    if audio_signal.dtype != 'float64':
        audio_signal = wav_to_float(audio_signal)

    return audio_signal, sample_rate


def load_wav(wav_path, desired_sample_rate=22050):

    sequence, sample_rate = read_wav(wav_path)
    sequence = ensure_sample_rate(sequence, desired_sample_rate, sample_rate)
    return np.transpose(sequence)


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


def load_tracks(dataset_path, tracks, include_mix=False):
    """
    if include_mix: return list of (mix, accomapny, voice). else, return list of (accomapny, voice)

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

        if include_mix:
            mix = accompany + vocal
            sequences.append((mix, accompany, vocal))
        else:
            sequences.append((accompany, vocal))

    return sequences


def write_wav(x, filename, sample_rate):

    if type(x) != np.ndarray:
        x = np.array(x)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sf.write(filename, x, sample_rate)


def slice_wav(path, save_path, start, end):

    os.system('ffmpeg -ss {} -t {} -i {} {}'.format(start, end, path, save_path))


def compute_input_length(input_shape, num_layers, enc_filter_size, dec_filter_size):

    x = input_shape
    for _ in range(num_layers):
        x = x + dec_filter_size - 1
        x = (x+1)/ 2

    x = np.asarray(np.ceil(x), dtype=np.int64)

    context_input_length = x
    context_output_length = x

    context_input_length = context_input_length + enc_filter_size - 1

    # Go from centre feature map through up- and downsampling blocks
    for i in range(num_layers):
        context_output_length = 2 * context_output_length - 1  # Upsampling
        context_output_length = context_output_length - dec_filter_size + 1  # Conv

        context_input_length = 2 * context_input_length - 1  # Decimation
        context_input_length = context_input_length + enc_filter_size - 1  # Conv

    print('context input, output lengths : {}, {}'.format(context_input_length, context_output_length))

    return context_input_length, context_output_length