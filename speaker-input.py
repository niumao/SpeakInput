#!/usr/bin/env python3
"""
Real-time voice recording and transcription using Faster Whisper with Silero VAD.
Optimized for CPU (AMD Ryzen 7 4800U).
"""

import pyaudio
import numpy as np
import wave
import tempfile
import os
import threading
import queue
import subprocess
import urllib.request
import onnxruntime as ort
from collections import deque
from faster_whisper import WhisperModel

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz

# VAD settings
VAD_FRAME_MS = 30           # Frame size for VAD (30ms)
VAD_FRAME_SAMPLES = int(RATE * VAD_FRAME_MS / 1000)  # 480 samples at 16kHz
VAD_THRESHOLD = 0.5         # Speech probability threshold
VAD_MIN_SPEECH_MS = 250     # Min speech duration to trigger recording
VAD_MIN_SILENCE_MS = 700    # Silence duration to end utterance
VAD_PRE_SPEECH_MS = 300     # Pre-buffer to capture word onsets

# Derived VAD constants (in frames)
VAD_MIN_SPEECH_FRAMES = VAD_MIN_SPEECH_MS // VAD_FRAME_MS
VAD_MIN_SILENCE_FRAMES = VAD_MIN_SILENCE_MS // VAD_FRAME_MS
VAD_PRE_SPEECH_FRAMES = VAD_PRE_SPEECH_MS // VAD_FRAME_MS

# Model settings (CPU optimized)
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large-v2
DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # int8 is faster on CPU

# Terminal colors
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"

# VAD model path
VAD_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "silero_vad.onnx")


def download_vad_model(model_path=VAD_MODEL_PATH):
    """Download Silero VAD ONNX model if not present."""
    if not os.path.exists(model_path):
        print(f"Downloading Silero VAD ONNX model to {model_path}...")
        url = "https://huggingface.co/onnx-community/silero-vad/resolve/main/model.onnx"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    return model_path


def type_text(text):
    """Type text to focused application via ydotool."""
    try:
        subprocess.run(['ydotool', 'type', '--', text], check=True)
    except FileNotFoundError:
        print("\nWarning: ydotool not found. Install with: sudo apt install ydotool")
    except subprocess.CalledProcessError as e:
        print(f"\nWarning: ydotool failed: {e}")


class RealtimeTranscriber:
    def __init__(self):
        print("Loading Faster Whisper model (this may take a moment on first run)...")
        print(f"Model: {MODEL_SIZE}, Device: {DEVICE}, Compute: {COMPUTE_TYPE}")

        self.model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=8,  # Adjust based on your CPU cores
        )

        print("Loading Silero VAD ONNX model...")
        model_path = download_vad_model()
        self.vad_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # # LSTM states for VAD (must persist between frames)
        # self.vad_h = np.zeros((2, 1, 64), dtype=np.float32)
        # self.vad_c = np.zeros((2, 1, 64), dtype=np.float32)
        self.vad_state = np.zeros((2, 1, 128), dtype=np.float32)

        self.audio_queue = queue.Queue()
        self.running = False
        self.current_status = None

    def update_status(self, status):
        """Update terminal with colored status."""
        if status == self.current_status:
            return

        self.current_status = status

        if status == "LISTENING":
            color = COLOR_GREEN
        elif status == "SPEAKING":
            color = COLOR_YELLOW
        elif status == "TRANSCRIBING":
            color = COLOR_CYAN
        else:
            color = COLOR_RESET

        # Clear line and print status
        print(f"\r{color}{status}{COLOR_RESET}                    ", end="", flush=True)

    def check_vad(self, audio_frame):
        """Get speech probability from VAD ONNX model."""
        audio_input = audio_frame.reshape(1, -1).astype(np.float32)
        sr_input = np.array([RATE], dtype=np.int64)

        # Run inference with the 'state' key
        outputs = self.vad_session.run(
            None,
            {
                'input': audio_input,
                'sr': sr_input,
                'state': self.vad_state, # Use 'state' instead of 'h' and 'c'
            }
        )

        # Update state for next frame
        speech_prob, self.vad_state = outputs

        return speech_prob[0, 0]

    def reset_vad_states(self):
        # """Reset VAD LSTM states (call when starting new utterance detection)."""
        # self.vad_h = np.zeros((2, 1, 64), dtype=np.float32)
        # self.vad_c = np.zeros((2, 1, 64), dtype=np.float32)
        self.vad_state = np.zeros((2, 1, 128), dtype=np.float32)

    def record_worker(self):
        """Continuously record audio with VAD-based utterance detection."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=VAD_FRAME_SAMPLES
        )

        print("\n" + "=" * 50)
        print("Real-time VAD-Based Voice Transcription")
        print("=" * 50)
        print("Press Ctrl+C to stop\n")

        # State machine variables
        is_speaking = False
        speech_frames = 0
        silence_frames = 0
        utterance_buffer = []

        # Pre-speech ring buffer (captures audio before speech is confirmed)
        pre_speech_buffer = deque(maxlen=VAD_PRE_SPEECH_FRAMES)

        self.update_status("LISTENING")

        while self.running:
            # Read one VAD frame
            data = stream.read(VAD_FRAME_SAMPLES, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # Check VAD
            speech_prob = self.check_vad(audio_frame)
            is_speech = speech_prob >= VAD_THRESHOLD

            if not is_speaking:
                # LISTENING state
                pre_speech_buffer.append(audio_frame)

                if is_speech:
                    speech_frames += 1
                    if speech_frames >= VAD_MIN_SPEECH_FRAMES:
                        # Transition to SPEAKING state
                        is_speaking = True
                        silence_frames = 0
                        # Add pre-speech buffer to capture word onset
                        utterance_buffer = list(pre_speech_buffer)
                        utterance_buffer.append(audio_frame)
                        self.update_status("SPEAKING")
                else:
                    speech_frames = 0
            else:
                # SPEAKING state
                utterance_buffer.append(audio_frame)

                if is_speech:
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames >= VAD_MIN_SILENCE_FRAMES:
                        # End of utterance - queue for transcription
                        if utterance_buffer:
                            full_audio = np.concatenate(utterance_buffer)
                            self.audio_queue.put(full_audio)

                        # Reset state
                        is_speaking = False
                        speech_frames = 0
                        silence_frames = 0
                        utterance_buffer = []
                        pre_speech_buffer.clear()
                        self.reset_vad_states()
                        self.update_status("LISTENING")

        # Handle any remaining audio
        if utterance_buffer:
            full_audio = np.concatenate(utterance_buffer)
            self.audio_queue.put(full_audio)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def transcribe_worker(self):
        """Continuously transcribe audio from queue."""
        while self.running or not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get(timeout=1)

                self.update_status("TRANSCRIBING")

                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(RATE)
                    wf.writeframes((audio_data * 32768).astype(np.int16).tobytes())

                # Transcribe (vad_filter=False since we already did VAD)
                segments, info = self.model.transcribe(
                    temp_file.name,
                    language="zh",  # Change to your language or None for auto-detect
                    beam_size=5,
                    vad_filter=False,
                )

                text = "".join(s.text for s in segments).strip()
                if text:
                    # Type text to focused application
                    type_text(text)
                    # Print on new line, then restore status
                    print(f"\r>>> {text}                              ")
                    self.current_status = None  # Force status update
                    self.update_status("LISTENING")

                os.unlink(temp_file.name)

            except queue.Empty:
                continue

    def run(self):
        self.running = True

        record_thread = threading.Thread(target=self.record_worker)
        transcribe_thread = threading.Thread(target=self.transcribe_worker)

        record_thread.start()
        transcribe_thread.start()

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.running = False

        record_thread.join()
        transcribe_thread.join()
        print("Exiting...")


def main():
    transcriber = RealtimeTranscriber()
    transcriber.run()


if __name__ == "__main__":
    main()

