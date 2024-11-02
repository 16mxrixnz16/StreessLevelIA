import torch
import torchaudio
from torchaudio import transforms
import random

class AudioData:
    def __init__(self, signal, sample_rate):
        self.signal = signal
        self.sample_rate = sample_rate
    
    def __str__(self):
        return f"AudioData(sample_rate={self.sample_rate}Hz, signal_shape={self.signal.shape})"

# Audio processing parameters
new_channel = 2  # Convert all audio to stereo
new_sr = 16000  # Target sample rate
max_ms = 10000  # Maximum audio length in milliseconds

def rechannel(audio_data: AudioData, new_channel: int) -> AudioData:
    """Convert audio to the desired number of channels"""
    if audio_data.signal.shape[0] == new_channel:
        return audio_data
    
    if new_channel == 1:
        resig = audio_data.signal[:1, :]
    else:
        resig = torch.cat([audio_data.signal, audio_data.signal])
    
    return AudioData(resig, audio_data.sample_rate)

def resample(audio_data: AudioData, newsr: int) -> AudioData:
    """Resample audio to the target sample rate"""
    if audio_data.sample_rate == newsr:
        return audio_data
    
    num_channels = audio_data.signal.shape[0]
    resig = torchaudio.transforms.Resample(audio_data.sample_rate, newsr)(audio_data.signal[:1, :])
    
    if num_channels > 1:
        retwo = torchaudio.transforms.Resample(audio_data.sample_rate, newsr)(audio_data.signal[1:, :])
        resig = torch.cat([resig, retwo])
    
    return AudioData(resig, newsr)

def pad_trunc(audio_data: AudioData, max_ms: int) -> AudioData:
    """Pad or truncate audio to the target length"""
    num_rows, sig_len = audio_data.signal.shape
    max_len = audio_data.sample_rate // 1000 * max_ms

    if sig_len > max_len:
        sig = audio_data.signal[:, :max_len]
    elif sig_len < max_len:
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))
        sig = torch.cat((pad_begin, audio_data.signal, pad_end), 1)
    else:
        sig = audio_data.signal
    
    return AudioData(sig, audio_data.sample_rate)

def process_audio(audio_data: AudioData) -> AudioData:
    """Process audio with all transformations"""
    # Convert to stereo
    audio_data = rechannel(audio_data, new_channel)
    
    # Resample to target sample rate
    audio_data = resample(audio_data, new_sr)
    
    # Pad or truncate to fixed length
    audio_data = pad_trunc(audio_data, max_ms)
    
    return audio_data

def open_audio_file(audio_path: str) -> AudioData:
    """Load an audio file and return AudioData object"""
    try:
        signal, sample_rate = torchaudio.load(audio_path, backend="soundfile")
        return AudioData(signal, sample_rate)
    except Exception as e:
        print(f"Error loading audio file: {audio_path}")
        print(f"Error details: {str(e)}")
        raise 