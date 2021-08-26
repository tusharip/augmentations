import torch
import torchaudio
import torchaudio.functional as F
import sounddevice as sd
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    ax1 = plt.subplot(111)
    ax1.plot(time_axis, waveform[0], linewidth=1)
    ax1.set_xlabel('time')
    plt.show()




def play_audio(waveform, sr):
    waveform = waveform.squeeze().numpy()
    sd.play(waveform, sr)
    sd.wait()




def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show()

if __name__ =="__main__":
    hamming(1, 50000)