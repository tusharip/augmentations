import torch
import torchaudio
import torchaudio.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import play_audio, plot_waveform, plot_specgram


def hann_window( m):
    n=np.arange(m)
    window = 0.54 - 0.46*np.cos(2*np.pi*n/(m-1))
    return [window]

def framing(waveform, sampling_rate, window_ms =20., stride_ms =10, window_fun=None):
    window_size =int(sampling_rate*window_ms*0.001)
    stride_size =int(sampling_rate*stride_ms*0.001)

    #extractig all windows
    windows = []

    for i in range(0, waveform.shape[-1], stride_size):
        frame = waveform[:,i:i+window_size]

        if frame.shape[-1] == window_size:
            windows.append(waveform[:,i:i+window_size])
    windows = torch.stack(windows, dim=0)
    plot_waveform(windows[0], sr)
    hann    = torch.tensor(window_fun(window_size)).unsqueeze(0)
    frames = windows*hann

    return frames


if __name__ == "__main__":
    a = torch.randn(10, 3, 24, 24)
    print(a.unsqueeze(0).shape)
    wave, sr = torchaudio.load("./data/voice.wav")
    print(wave.shape, sr)

    framing(wave, sr, window_fun=hann_window)
    # print_stats(wave, sample_rate=sr)
    # plot_waveform(wave, sr)

    exit()
    plot_specgram(wave, sr)
    play_audio(wave, sr)