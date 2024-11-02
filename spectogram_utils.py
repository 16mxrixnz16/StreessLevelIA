import torch
import torchaudio
from torchaudio import transforms
from audio_utils import AudioData

def create_spectogram(audio_data: AudioData, n_mels=64, n_fft=1024, hop_len=None) -> torch.Tensor:
    """
    Create a mel spectogram from audio data
    
    Args:
        audio_data: AudioData object containing signal and sample rate
        n_mels: Number of mel filterbanks
        n_fft: Size of FFT
        hop_len: Length of hop between STFT windows
        
    Returns:
        torch.Tensor: Mel spectogram in decibel scale
    """
    top_db = 80

    # Compute the Mel spectrogram
    spec = transforms.MelSpectrogram(
        sample_rate=audio_data.sample_rate,
        n_fft=n_fft,
        hop_length=hop_len,
        n_mels=n_mels
    )(audio_data.signal)

    # Convert the spectrogram to decibel scale
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)

    return spec 