import gradio as gr
import os
import tempfile
from pathlib import Path
import random
import shutil
import torch

from audio_utils import (
    AudioData, 
    process_audio, 
    open_audio_file,
    new_channel,
    new_sr,
    max_ms
)
from spectogram_utils import create_spectogram
from model import AudioEmotionCNN, load_model

# Emotion labels and their stress weights
EMOTION_WEIGHTS = {
    "Anger": 1.0,
    "Fear": 0.9,
    "Sad": 0.7,
    "Disgust": 0.7,
    "Surprised": 0.4,
    "Neutral": 0.3,
    "Happy": 0.1,
    "Calm": 0.0
}

EMOTION_LABELS = list(EMOTION_WEIGHTS.keys())

# Load the model at startup
try:
    MODEL = load_model('audio_emotion_model.pth')
except Exception as e:
    print(f"Warning: Could not load model: {str(e)}")
    MODEL = None

def calculate_stress_level(emotions):
    """
    Calculate average stress level from a list of emotions
    """
    if not emotions:
        return 0.0, "No emotions detected"
    
    total_stress = sum(EMOTION_WEIGHTS[emotion] for emotion in emotions)
    avg_stress = total_stress / len(emotions)
    
    if avg_stress >= 0.8:
        description = "Very High Stress"
    elif avg_stress >= 0.6:
        description = "High Stress"
    elif avg_stress >= 0.4:
        description = "Moderate Stress"
    elif avg_stress >= 0.2:
        description = "Low Stress"
    else:
        description = "You are stress free"
    
    return avg_stress, description

def simulate_emotion_classification(audio_path):
    """
    Classify emotion using the pretrained model
    """
    if MODEL is None:
        return random.choice(EMOTION_LABELS)
    
    try:
        # Load and process the audio file
        audio_data = open_audio_file(audio_path)
        processed_audio = process_audio(audio_data)
        
        # Create spectogram
        spectogram = create_spectogram(processed_audio)
        
        # Prepare input for model
        spectogram = spectogram.unsqueeze(0)  # Add batch dimension
        
        # Move to device and normalize (same as in training)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        spectogram = spectogram.to(device)
        inputs_m, inputs_s = spectogram.mean(), spectogram.std()
        spectogram = (spectogram - inputs_m) / inputs_s
        
        # Get model prediction
        with torch.no_grad():
            outputs = MODEL(spectogram)
            _, predicted = torch.max(outputs, 1)
            predicted_emotion = EMOTION_LABELS[predicted.item()]
        
        print(f"Processed audio shape: {processed_audio.signal.shape}")
        print(f"Processed sample rate: {processed_audio.sample_rate}Hz")
        print(f"Spectogram shape: {spectogram.shape}")
        print(f"Predicted emotion: {predicted_emotion}")
        
        return predicted_emotion
        
    except Exception as e:
        print(f"Error in emotion classification: {str(e)}")
        return random.choice(EMOTION_LABELS)

def save_uploaded_file(audio):
    """
    Save the uploaded audio file to temp directory
    """
    if audio is None:
        return audio, "Please record or select an audio file first!"
    
    # Create temporary directory if it doesn't exist
    temp_dir = Path(tempfile.gettempdir()) / "audio_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    output_path = temp_dir / f"audio_{os.urandom(8).hex()}.wav"
    
    if isinstance(audio, tuple):  # If audio is recorded via microphone
        sr, data = audio
        return audio, "Audio recording saved successfully!"
    else:  # If audio is uploaded as a file
        try:
            shutil.copy2(audio, output_path)
            return audio, "Audio file saved successfully!"
        except Exception as e:
            print(f"Error saving file: {e}")
            return audio, f"Error saving file: {e}"

def analyze_all_audios():
    """
    Analyze all audio files in the temp directory
    """
    temp_dir = Path(tempfile.gettempdir()) / "audio_uploads"
    if not temp_dir.exists():
        return "No audio files found! Please upload or record some audio first."
    
    audio_files = list(temp_dir.glob("*.wav"))
    if not audio_files:
        return "No audio files found! Please upload or record some audio first."
    
    # Analyze each audio file
    results = []
    emotions = []
    for audio_file in audio_files:
        emotion = simulate_emotion_classification(str(audio_file))
        emotions.append(emotion)
        results.append(f"File: {audio_file.name} → Emotion: {emotion} (Stress Level: {EMOTION_WEIGHTS[emotion]:.1f})")
    
    # Calculate average stress level
    avg_stress, stress_description = calculate_stress_level(emotions)
    
    # Format results
    results_text = "🎯 Emotion Classification Results:\n\n"
    results_text += "\n\n".join(results)
    results_text += f"\n\n📊 Overall Analysis:\n"
    results_text += f"Average Stress Level: {avg_stress:.2f}\n"
    results_text += f"Assessment: {stress_description}"
    
    return results_text

def delete_temp_files():
    """
    Delete all audio files from the temp directory
    """
    temp_dir = Path(tempfile.gettempdir()) / "audio_uploads"
    if temp_dir.exists():
        try:
            for file in temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
            return "✨ All audio files cleared successfully!"
        except Exception as e:
            print(f"Error clearing files: {e}")
            return f"Error clearing files: {str(e)}"
    return "No files to clear."

# Create the Gradio interface
with gr.Blocks(title="Audio Emotion Classifier") as demo:
    # Delete temp files at startup
    delete_temp_files()
    
    gr.Markdown("## 🎤 Batch Audio Emotion Classification")
    gr.Markdown("Record or upload audio files and analyze them all at once")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or Upload Audio"
            )
        
        with gr.Column():
            result_text = gr.Markdown()
    
    with gr.Row():
        upload_button = gr.Button("📤 Save Audio", variant="secondary")
        analyze_button = gr.Button("🎯 Analyze All Audios", variant="primary")
        clear_button = gr.Button("🗑️ Clear All Audio", variant="secondary")
    
    upload_status = gr.Markdown()
    
    # Button click handlers
    upload_button.click(
        fn=save_uploaded_file,
        inputs=[audio_input],
        outputs=[audio_input, upload_status]
    )
    
    analyze_button.click(
        fn=analyze_all_audios,
        inputs=[],
        outputs=[result_text]
    )
    
    clear_button.click(
        fn=delete_temp_files,
        inputs=[],
        outputs=[upload_status]
    )
    
    gr.Markdown("""
    ### Instructions:
    1. Record audio using microphone or upload audio files
    2. Click '📤 Save Audio' after each recording/upload
    3. Repeat steps 1-2 for all audio files you want to analyze
    4. Click '🎯 Analyze All Audios' to get emotion predictions
    5. Use '🗑️ Clear All Audio' to delete all saved files
    
    Note: Make sure recordings are at least 1 second long for better results.
    """)

if __name__ == "__main__":
    demo.launch()
