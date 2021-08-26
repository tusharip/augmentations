import torch
import torchaudio
import torchaudio.functional as F

from utils import play_audio, plot_waveform, plot_specgram






if __name__ == "__main__":
    wave, sr = torchaudio.load("./data/voice.wav")
    print(wave.shape, sr)
    # print_stats(wave, sample_rate=sr)
    plot_waveform(wave, sr)

    exit()
    plot_specgram(wave, sr)
    play_audio(wave, sr)