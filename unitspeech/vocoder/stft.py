"""
BSD 3-Clause License
Copyright (c) 2017, Prem Seetharaman
All rights reserved.
* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import math
import librosa
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn
import librosa.util as librosa_util


def label_2_float(x, bits):
    return 2 * x / (2 ** bits - 1.0) - 1.0


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.0) * (2 ** bits - 1) / 2
    return x.clip(0, 2 ** bits - 1)


def load_wav(path, hp):
    return librosa.load(path, sr=hp.audio.sample_rate)[0]


def save_wav(x, path, hp):
    librosa.output.write_wav(path, x.astype(np.float32), sr=hp.audio.sample_rate)


def split_signal(x):
    unsigned = x + 2 ** 15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2 ** 15


def encode_16bits(x):
    return np.clip(x * 2 ** 15, -(2 ** 15), 2 ** 15 - 1).astype(np.int16)


mel_basis = None


def energy(y):
    # Extract energy
    S = librosa.magphase(stft(y))[0]
    e = np.sqrt(np.sum(S ** 2, axis=0))  # np.linalg.norm(S, axis=0)
    return e.squeeze()  # (Number of frames) => (654,)


def pitch(y, hp):
    # Extract Pitch/f0 from raw waveform using PyWORLD
    y = y.astype(np.float64)
    """
    f0_floor : float
        Lower F0 limit in Hz.
        Default: 71.0
    f0_ceil : float
        Upper F0 limit in Hz.
        Default: 800.0
    """
    f0, timeaxis = pw.dio(
        y,
        hp.audio.sample_rate,
        frame_period=hp.audio.hop_length / hp.audio.sample_rate * 1000,
    )  # For hop size 256 frame period is 11.6 ms
    return f0  # (Number of Frames) = (654,)


def linear_to_mel(spectrogram, hp):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis(hp)
    return np.dot(mel_basis, spectrogram)


def build_mel_basis(hp):
    return librosa.filters.mel(
        hp.audio.sample_rate,
        hp.audio.n_fft,
        n_mels=hp.audio.num_mels,
        fmin=hp.audio.fmin,
    )


def normalize(S, hp):
    return np.clip((S - hp.audio.min_level_db) / -hp.audio.min_level_db, 0, 1)


def denormalize(S, hp):
    return (np.clip(S, 0, 1) * -hp.audio.min_level_db) + hp.audio.min_level_db


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def spectrogram(y, hp):
    D = stft(y, hp)
    S = amp_to_db(np.abs(D)) - hp.audio.ref_level_db
    return normalize(S, hp)


def melspectrogram(y, hp):
    D = stft(y, hp)
    S = amp_to_db(linear_to_mel(np.abs(D), hp))
    return normalize(S, hp)


def stft(y, hp):
    return librosa.stft(
        y=y,
        n_fft=hp.audio.n_fft,
        hop_length=hp.audio.hop_length,
        win_length=hp.audio.win_length,
    )


def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    # TODO : get rid of log2 - makes no sense
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def reconstruct_waveform(mel, hp, n_iter=32):
    """Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform."""
    denormalized = denormalize(mel)
    amp_mel = db_to_amp(denormalized)
    S = librosa.feature.inverse.mel_to_stft(
        amp_mel,
        power=1,
        sr=hp.audio.sample_rate,
        n_fft=hp.audio.n_fft,
        fmin=hp.audio.fmin,
    )
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter, hop_length=hp.audio.hop_length, win_length=hp.audio.win_length
    )
    return wav


def quantize_input(input, min, max, num_bins=256):
    bins = np.linspace(min, max, num=num_bins)
    quantize = np.digitize(input, bins)
    return quantize


def window_sumsquare(
    window,
    n_frames,
    hop_length=200,
    win_length=800,
    n_fft=800,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles).cuda())
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(
        self, filter_length=800, hop_length=200, win_length=800, window="hann"
    ):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        # https://github.com/NVIDIA/tacotron2/issues/125
        forward_transform = F.conv1d(
            input_data.cuda(),
            Variable(self.forward_basis, requires_grad=False).cuda(),
            stride=self.hop_length,
            padding=0,
        ).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        sampling_rate=22050,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output, magnitudes
