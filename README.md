# Real-Time VAD Voice Transcriber for Linux



This tool is a high-performance, local **voice-to-text utility** designed to provide hands-free dictation for Linux systems. It functions as a virtual keyboard, allowing you to input text into any focused application using only your voice.



---



## 🎯 Purpose

The script serves as a **private, system-wide voice input method**. Its primary objectives include:

* **System-Wide Dictation**: It utilizes `ydotool` to "type" transcribed text directly into the active window, such as a browser, document editor, or terminal.

* **Privacy & Local Processing**: All audio stays on your machine; transcription is performed locally via `faster-whisper` and `silero-vad`, ensuring data is not sent to external servers.

* **Hardware Optimization**: Specifically tuned for CPU execution (e.g., AMD Ryzen 7 4800U) using `int8` quantization for near-instant results.

* **Hands-Free Operation**: Uses Silero Voice Activity Detection (VAD) to automatically detect when you start and stop speaking, removing the need for a "Push-to-Talk" button.



---



## 🚀 Features

* **Intelligent VAD**: Distinguishes between human speech and background noise to avoid unnecessary recordings.

* **Pre-speech Buffering**: Maintains a 300ms buffer to ensure the beginning of your sentence is never cut off.

* **Multi-threaded Architecture**: Separate worker threads handle recording and transcription to ensure no audio frames are dropped.

* **Visual Feedback**: Real-time terminal status updates (LISTENING, SPEAKING, TRANSCRIBING) using ANSI colors.



---



## 🛠 Prerequisites



### System Dependencies

The tool requires `ydotool` to simulate keyboard input and `portaudio` for recording.

```bash

# Ubuntu/Debian example

sudo apt update

sudo apt install ydotool portaudio19-dev



```



### Python Environment



Install the following libraries to support the transcription engine and VAD model:



```bash

pip install numpy pyaudio onnxruntime faster-whisper



```



---



## 🚦 Getting Started



1. **Launch the Input Daemon**: `ydotool` requires its daemon to be running to simulate keypresses.

```bash

sudo ydotoold &



```





2. **Run the Script**:

```bash

python tes.py



```





3. **Usage**: The terminal will display **LISTENING** in green. Speak clearly; once you stop, the tool will automatically transcribe and type the text into your focused app.



---



## ⚙️ Technical Configuration



You can modify these constants at the top of the script to match your hardware:



| Setting | Description | Default |

| --- | --- | --- |

| `MODEL_SIZE` | Whisper model size (tiny, base, small, medium, large) | `base` |

| `COMPUTE_TYPE` | Weight precision (`int8` is fastest for CPU) | `int8` |

| `VAD_THRESHOLD` | Sensitivity of speech detection (0.0 to 1.0) | `0.5` |

| `language` | Target language for transcription (e.g., "en", "zh") | `"zh"` |


