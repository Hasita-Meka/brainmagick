"""Step 4.3: one-channel log envelope (same definition as the Envelope target) and its slow version."""
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt

SR_AUDIO = 16_000
N_FFT, HOP = 512, 128   # the paper's mel settings (Sec. 2.4): 125 frames/s at 16 kHz
SR_FEAT = 120           # brain and feature rate
EPS = 1e-5              # the paper's log floor
LP_HZ = 8.0             # slow envelope for similarity (plan, Change 3)


def load_wav(path) -> tuple:
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    return (x.mean(axis=1) if x.ndim == 2 else x), sr


def log_envelope(wav: np.ndarray, sr: int) -> np.ndarray:
    """Log broadband power of a whole sound file at 120 Hz.
    Hann window, centred frames, power summed over all frequency bins; the constant scale of
    the STFT normalization only shifts the log and disappears after standardization."""
    if sr != SR_AUDIO:
        g = np.gcd(sr, SR_AUDIO)
        wav = resample_poly(wav, SR_AUDIO // g, sr // g)
    win = np.hanning(N_FFT + 1)[:-1]                       # periodic Hann, like torch.hann_window
    x = np.pad(wav, N_FFT // 2, mode="reflect")            # center=True framing, like torch.stft
    n_frames = 1 + (len(x) - N_FFT) // HOP
    frames = x[np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]] * win
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).sum(axis=1) / (win ** 2).sum()
    return resample_poly(np.log(EPS + power), 24, 25)      # 125 Hz -> 120 Hz


def slow(env: np.ndarray, cutoff: float = LP_HZ, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass. Filter the continuous envelope, then slice, to avoid edge effects."""
    sos = butter(order, cutoff, btype="low", fs=SR_FEAT, output="sos")
    return sosfiltfilt(sos, env)


def slice_clips(env_by_seg: dict, pool, dur: float = 3.0) -> np.ndarray:
    n = int(round(dur * SR_FEAT))
    out = np.empty((len(pool), n), np.float32)
    for k, (s, a) in enumerate(zip(pool.seg, pool.start)):
        i0 = int(round(a * SR_FEAT))
        seg = env_by_seg[s]
        piece = seg[i0:i0 + n]
        if len(piece) < n:   # last frame rounding at a file end
            piece = np.pad(piece, (0, n - len(piece)), mode="edge")
        out[k] = piece
    return out
