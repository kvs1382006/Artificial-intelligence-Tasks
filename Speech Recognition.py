import speech_recognition as sr
import os

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes a short audio file into text using SpeechRecognition library.
    """
    recognizer = sr.Recognizer()
    
    if not os.path.exists(audio_file_path):
        return f"Error: File '{audio_file_path}' not found."
        
    with sr.AudioFile(audio_file_path) as source:
        print("[INFO] Loading audio file and reducing ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio_data = recognizer.record(source)
        
    try:
        print("[INFO] Transcribing audio...")
        transcription = recognizer.recognize_google(audio_data)
        return transcription
    except sr.UnknownValueError:
        return "Speech Recognition could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from service; {e}"

if __name__ == "__main__":
    # Provide path to a local WAV file
    sample_audio = "sample_audio.wav"
    
    print("=" * 60)
    print("SPEECH RECOGNITION TRANSCRIBER")
    print("=" * 60)
    
    # Run transcription
    result = transcribe_audio(sample_audio)
    print(f"\nTranscription Result:\n{result}")