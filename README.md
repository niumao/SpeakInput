This version of the README is specifically written to highlight that your tool is **100% free, runs entirely on your CPU, and requires no internet or API keys.**

---

# Local Voice-to-Text for Linux (CPU Optimized)

A high-performance, **completely local** voice-typing utility. This tool turns your voice into keyboard input without sending any data to the cloud. It is specifically optimized to run smoothly on standard laptop/desktop CPUs without needing a dedicated GPU.

---

## 💎 Why This Tool?

* **💰 Zero Cost:** No OpenAI API fees or monthly subscriptions. It uses open-source models that run on your own hardware.
* **💻 CPU Optimized:** Specifically tuned for processors (like AMD Ryzen or Intel Core series) using `int8` quantization. You don't need a high-end graphics card.
* **🔒 100% Private:** Your audio never leaves your computer. It works entirely offline, making it safe for sensitive work.
* **⌨️ System-Wide Input:** Unlike web-based transcribers, this "types" directly into any app (Chrome, VS Code, Discord, Slack, etc.) using `ydotool`.

---

## 🚀 Key Features

* **Faster-Whisper Engine:** Uses a highly optimized version of OpenAI's Whisper for near-instant transcription.
* **Silero VAD (Voice Activity Detection):** Smart enough to ignore background noise and only "listen" when you are actually speaking.
* **Auto-Typing:** Automatically injects text into the active window once you finish a sentence.
* **Pre-Speech Buffer:** Captures the split second *before* you start talking so the first word of your sentence is never cut off.

---

## 🛠 Installation

### 1. System Dependencies

You need `portaudio` for the microphone and `ydotool` to handle the typing.

```bash
# Ubuntu / Debian / Mint
sudo apt update
sudo apt install ydotool portaudio19-dev

```

### 2. Python Setup

Install the local AI engine and audio libraries:

```bash
pip install numpy pyaudio onnxruntime faster-whisper

```

---

## 🚦 Usage

1. **Start the typing daemon:**
`ydotool` requires its background service to be running.
```bash
sudo ydotoold &

```


2. **Run the tool:**
```bash
python speaker-input.py

```


3. **How to use:**
* The terminal will show **LISTENING** in green.
* Just start talking. The tool detects your voice automatically (**SPEAKING**).
* Stop talking for a moment, and it will automatically transcribe (**TRANSCRIBING**) and "type" the text into your focused window.



---

## ⚙️ Configuration (Edit `speaker-input.py`)

| Setting | Description | Recommended for CPU |
| --- | --- | --- |
| `MODEL_SIZE` | AI Model size (`tiny`, `base`, `small`) | `base` (best balance) |
| `COMPUTE_TYPE` | Math precision | `int8` (fastest for CPU) |
| `language` | Transcription language | `"zh"` (Chinese) or `"en"` (English) |

---
