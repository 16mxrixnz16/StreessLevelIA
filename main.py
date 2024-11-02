import gradio as gr
import os
import tempfile
from pathlib import Path
import random
import shutil

# Simulated emotion labels
EMOTION_LABELS = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Calm", "Surprised"]

def simulate_emotion_classification(audio_path):
    """
    Placeholder function to simulate emotion classification.
    """
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
            print(f"Error saving file: {e}")  # Debug print
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
    for audio_file in audio_files:
        emotion = simulate_emotion_classification(str(audio_file))
        results.append(f"File: {audio_file.name} → Emotion: {emotion}")
    
    # Format results
    results_text = "🎯 Emotion Classification Results:\n\n"
    results_text += "\n\n".join(results)
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
            print(f"Error clearing files: {e}")  # Debug print
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
