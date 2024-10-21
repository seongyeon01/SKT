import torchaudio
import random
import torch
from torchaudio.transforms import Resample, Spectrogram

def load_and_resample(file_path, target_sample_rate):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0).unsqueeze(0)
    return waveform

def extract_sec(waveform, sr, sec = 1, type = 'r', start = None):
    if start is not None:
        return waveform[:,start*sr:sr*(start+sec)]
    elif type == 'f': # front
        return waveform[:,:sr*sec]
    elif type == 'b': # back
        return waveform[:,-sr*sec:]
    elif type == 'r': # random
        s = int(random.uniform(0,waveform.shape[1]-sec*sr))
        return waveform[:,s:s+(sr*sec)]

def create_spectrogram(waveform, sample_rate, freq_range, n_fft=512, hop_length=300):
    spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length)
    spectrogram = spectrogram_transform(waveform)
    freq_bin = torch.linspace(0, sample_rate // 2, spectrogram.size(1))
    freq_mask = (freq_bin >= freq_range[0]) & (freq_bin <= freq_range[1])
    return spectrogram[:, freq_mask, :]

# STFT 추출 함수
def extract_stft(waveform, n_fft, hop_length, window):
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True)
    return stft

# STFT에서 주파수 범위를 필터링하는 함수
def filter_frequency_range(stft, sample_rate, freq_range, n_fft):
    freq_bin = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1)  # FFT 주파수 축 계산
    freq_mask = (freq_bin >= freq_range[0]) & (freq_bin <= freq_range[1])  # 주파수 범위 마스크
    return stft[:, freq_mask, :]  # 주파수 범위에 맞춰 필터링


def preprocess(sample, n_fft, hop_length, hann_window, sr, freq_range):
    sample_stft = extract_stft(sample, n_fft, hop_length, hann_window)
    sample_filter = filter_frequency_range(sample_stft, sr, freq_range, n_fft).squeeze()
    return sample_filter

def forpred(filtered_stft):
    data = torch.ones(1, 513, 64)
    c, h, w = filtered_stft.shape
    data[:c,:h,:w] = filtered_stft
    return data