from genericpath import isdir
import io
import os
import math
from re import sub
import tarfile
import multiprocessing
from librosa.core.audio import load

import scipy
import librosa
import requests
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
from IPython.display import Audio, display
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from pprint import pprint
# from dataset.dataset_utils import load_content_from_txt



def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
      ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def get_speech_sample(*, resample=None):
  return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)

def get_sample(*, resample=None):
  return _get_sample(SAMPLE_WAV_PATH, resample=resample)

def get_rir_sample(*, resample=None, processed=False):
  rir_raw, sample_rate = _get_sample(SAMPLE_RIR_PATH, resample=resample)
  if not processed:
    return rir_raw, sample_rate
  rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
  rir = rir / torch.norm(rir, p=2)
  rir = torch.flip(rir, [1])
  return rir, sample_rate

def get_noise_sample(*, resample=None):
  return _get_sample(SAMPLE_NOISE_PATH, resample=resample)

def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

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
  plt.show(block=False)

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

def inspect_file(path):
  print("-" * 10)
  print("Source:", path)
  print("-" * 10)
  print(f" - File size: {os.path.getsize(path)} bytes")
  print(f" - {torchaudio.info(path)}")

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

def plot_mel_fbank(fbank, title=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Filter bank')
  axs.imshow(fbank, aspect='auto')
  axs.set_ylabel('frequency bin')
  axs.set_xlabel('mel bin')
  plt.show(block=False)

def get_spectrogram(
    n_fft = 400,
    win_len = None,
    hop_len = None,
    power = 2.0,
):
  waveform, _ = get_speech_sample()
  spectrogram = T.Spectrogram(
      n_fft=n_fft,
      win_length=win_len,
      hop_length=hop_len,
      center=True,
      pad_mode="reflect",
      power=power,
  )
  return spectrogram(waveform)

def plot_pitch(waveform, sample_rate, pitch):
  figure, axis = plt.subplots(1, 1)
  axis.set_title("Pitch Feature")
  axis.grid(True)

  end_time = waveform.shape[1] / sample_rate
  time_axis = torch.linspace(0, end_time,  waveform.shape[1])
  axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

  axis2 = axis.twinx()
  time_axis = torch.linspace(0, end_time, pitch.shape[1])
  ln2 = axis2.plot(
      time_axis, pitch[0], linewidth=2, label='Pitch', color='green')

  axis2.legend(loc=0)
  plt.show(block=False)

def plot_kaldi_pitch(waveform, sample_rate, pitch, nfcc):
  figure, axis = plt.subplots(1, 1)
  axis.set_title("Kaldi Pitch Feature")
  axis.grid(True)

  end_time = waveform.shape[1] / sample_rate
  time_axis = torch.linspace(0, end_time,  waveform.shape[1])
  axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

  time_axis = torch.linspace(0, end_time, pitch.shape[1])
  ln1 = axis.plot(time_axis, pitch[0], linewidth=2, label='Pitch', color='green')
  axis.set_ylim((-1.3, 1.3))

  axis2 = axis.twinx()
  time_axis = torch.linspace(0, end_time, nfcc.shape[1])
  ln2 = axis2.plot(
      time_axis, nfcc[0], linewidth=2, label='NFCC', color='blue', linestyle='--')

  lns = ln1 + ln2
  labels = [l.get_label() for l in lns]
  axis.legend(lns, labels, loc=0)
  plt.show(block=False)

DEFAULT_OFFSET = 201
SWEEP_MAX_SAMPLE_RATE = 48000
DEFAULT_LOWPASS_FILTER_WIDTH = 6
DEFAULT_ROLLOFF = 0.99
DEFAULT_RESAMPLING_METHOD = 'sinc_interpolation'


def _get_log_freq(sample_rate, max_sweep_rate, offset):
  """Get freqs evenly spaced out in log-scale, between [0, max_sweep_rate // 2]

  offset is used to avoid negative infinity `log(offset + x)`.

  """
  half = sample_rate // 2
  start, stop = math.log(offset), math.log(offset + max_sweep_rate // 2)
  return torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double)) - offset

def _get_inverse_log_freq(freq, sample_rate, offset):
  """Find the time where the given frequency is given by _get_log_freq"""
  half = sample_rate // 2
  return sample_rate * (math.log(1 + freq / offset) / math.log(1 + half / offset))

def _get_freq_ticks(sample_rate, offset, f_max):
  # Given the original sample rate used for generating the sweep,
  # find the x-axis value where the log-scale major frequency values fall in
  time, freq = [], []
  for exp in range(2, 5):
    for v in range(1, 10):
      f = v * 10 ** exp
      if f < sample_rate // 2:
        t = _get_inverse_log_freq(f, sample_rate, offset) / sample_rate
        time.append(t)
        freq.append(f)
  t_max = _get_inverse_log_freq(f_max, sample_rate, offset) / sample_rate
  time.append(t_max)
  freq.append(f_max)
  return time, freq

def plot_sweep(waveform, sample_rate, title, max_sweep_rate=SWEEP_MAX_SAMPLE_RATE, offset=DEFAULT_OFFSET):
  x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
  y_ticks = [1000, 5000, 10000, 20000, sample_rate//2]

  time, freq = _get_freq_ticks(max_sweep_rate, offset, sample_rate // 2)
  freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
  freq_y = [f for f in freq if f >= 1000 and f in y_ticks and f <= sample_rate // 2]

  figure, axis = plt.subplots(1, 1)
  axis.specgram(waveform[0].numpy(), Fs=sample_rate)
  plt.xticks(time, freq_x)
  plt.yticks(freq_y, freq_y)
  axis.set_xlabel('Original Signal Frequency (Hz, log scale)')
  axis.set_ylabel('Waveform Frequency (Hz)')
  axis.xaxis.grid(True, alpha=0.67)
  axis.yaxis.grid(True, alpha=0.67)
  figure.suptitle(f'{title} (sample rate: {sample_rate} Hz)')
  plt.show(block=True)

def get_sine_sweep(sample_rate, offset=DEFAULT_OFFSET):
    max_sweep_rate = sample_rate
    freq = _get_log_freq(sample_rate, max_sweep_rate, offset)
    delta = 2 * math.pi * freq / sample_rate
    cummulative = torch.cumsum(delta, dim=0)
    signal = torch.sin(cummulative).unsqueeze(dim=0)
    return signal

def benchmark_resample(
    method,
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=DEFAULT_LOWPASS_FILTER_WIDTH,
    rolloff=DEFAULT_ROLLOFF,
    resampling_method=DEFAULT_RESAMPLING_METHOD,
    beta=None,
    librosa_type=None,
    iters=5
):
  if method == "functional":
    begin = time.time()
    for _ in range(iters):
      F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                 rolloff=rolloff, resampling_method=resampling_method)
    elapsed = time.time() - begin
    return elapsed / iters
  elif method == "transforms":
    resampler = T.Resample(sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                           rolloff=rolloff, resampling_method=resampling_method, dtype=waveform.dtype)
    begin = time.time()
    for _ in range(iters):
      resampler(waveform)
    elapsed = time.time() - begin
    return elapsed / iters
  elif method == "librosa":
    waveform_np = waveform.squeeze().numpy()
    begin = time.time()
    for _ in range(iters):
      librosa.resample(waveform_np, sample_rate, resample_rate, res_type=librosa_type)
    elapsed = time.time() - begin
    return elapsed / iters


def resample(
    method,
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=6,
    rolloff=0.99,
    resampling_method='sinc_interpolation',
    librosa_type=None,
):
    if sample_rate == resample_rate:
        return waveform
        
    if method == "functional":
        resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                    rolloff=rolloff, resampling_method=resampling_method)
    elif method == "transforms":
        resampler = T.Resample(sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                            rolloff=rolloff, resampling_method=resampling_method)
        resampled_waveform = resampler(waveform)
    elif method == "librosa":
        waveform_np = waveform.squeeze().numpy()
        resampled_waveform = librosa.resample(waveform_np, sample_rate, resample_rate, res_type=librosa_type)
    return resampled_waveform


def reample_test():
    load_path = rf'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset\1'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Kaggle_resample_16k\1'
    file_list = os.listdir(load_path)
    total_mse = 0

    os.chdir(load_path)
    start = time.time()
    for idx, f in enumerate(file_list):
        print(f'{idx+1}/{len(file_list)} reampling')
        mse = resample_audio(f, save_path, resample_rate=16000)
        total_mse += mse
    end = time.time()
    print(f"MSE: {total_mse}, Time: {end-start}")

def resample_audio(load_path, save_path, resample_rate, save_flag=False):
    waveform, sr = librosa.load(load_path, sr=None)
    resampled_waveform, _ = librosa.load(load_path, resample_rate)
    resampled_waveform = torch.from_numpy(np.array(np.float32(resampled_waveform)))
    waveform2 = resample('transforms', resampled_waveform, resample_rate, sr)
    mse = torch.square(waveform2 - waveform).mean().item()
    if save_flag:
        torchaudio.save(os.path.join(save_path, f'reampled_{resample_rate}_{os.path.basename(load_path)}'), resampled_waveform, resample_rate)
    return mse

def resample_audio2(load_path, save_path, resample_rate, save_flag=False):
    waveform, sr = torchaudio.load(load_path)
    resampled_waveform = resample('transforms', waveform, sr, resample_rate)
    waveform2 = resample('transforms', resampled_waveform, resample_rate, sr)
    mse = torch.square(waveform2 - waveform).mean().item()
    if save_flag:
        torchaudio.save(os.path.join(save_path, f'reampled_{resample_rate}_{os.path.basename(load_path)}'), resampled_waveform, resample_rate)
    return mse

def pytorch_resample_example():
    YESNO_DOWNLOAD_PROCESS.start()
    configs = {
        "downsample (48 -> 44.1 kHz)": [48000, 44100],
        "downsample (16 -> 8 kHz)": [16000, 8000],
        "upsample (44.1 -> 48 kHz)": [44100, 48000],
        "upsample (8 -> 16 kHz)": [8000, 16000],
    }

    for label in configs:
        times, rows = [], []
        sample_rate = configs[label][0]
        resample_rate = configs[label][1]
        waveform = get_sine_sweep(sample_rate)

        # sinc 64 zero-crossings
        f_time = benchmark_resample("functional", waveform, sample_rate, resample_rate, lowpass_filter_width=64)
        t_time = benchmark_resample("transforms", waveform, sample_rate, resample_rate, lowpass_filter_width=64)
        times.append([None, 1000 * f_time, 1000 * t_time])
        rows.append(f"sinc (width 64)")

        # sinc 6 zero-crossings
        f_time = benchmark_resample("functional", waveform, sample_rate, resample_rate, lowpass_filter_width=16)
        t_time = benchmark_resample("transforms", waveform, sample_rate, resample_rate, lowpass_filter_width=16)
        times.append([None, 1000 * f_time, 1000 * t_time])
        rows.append(f"sinc (width 16)")

        # kaiser best
        lib_time = benchmark_resample("librosa", waveform, sample_rate, resample_rate, librosa_type="kaiser_best")
        f_time = benchmark_resample(
            "functional",
            waveform,
            sample_rate,
            resample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="kaiser_window",
            beta=14.769656459379492)
        t_time = benchmark_resample(
            "transforms",
            waveform,
            sample_rate,
            resample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="kaiser_window",
            beta=14.769656459379492)
        times.append([1000 * lib_time, 1000 * f_time, 1000 * t_time])
        rows.append(f"kaiser_best")

        # kaiser fast
        lib_time = benchmark_resample("librosa", waveform, sample_rate, resample_rate, librosa_type="kaiser_fast")
        f_time = benchmark_resample(
            "functional",
            waveform,
            sample_rate,
            resample_rate,
            lowpass_filter_width=16,
            rolloff=0.85,
            resampling_method="kaiser_window",
            beta=8.555504641634386)
        t_time = benchmark_resample(
            "transforms",
            waveform,
            sample_rate,
            resample_rate,
            lowpass_filter_width=16,
            rolloff=0.85,
            resampling_method="kaiser_window",
            beta=8.555504641634386)
        times.append([1000 * lib_time, 1000 * f_time, 1000 * t_time])
        rows.append(f"kaiser_fast")

        df = pd.DataFrame(times,
                            columns=["librosa", "functional", "transforms"],
                            index=rows)
        df.columns = pd.MultiIndex.from_product([[f"{label} time (ms)"],df.columns])
        display(df.round(2))


if __name__ == '__main__':
    # pytorch_resample_example()
    # reample_test()
    pass
