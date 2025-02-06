# Spot: Cognitive Companion AI Client 🚀

A speech-to-speech AI client for the Boston Dynamics Spot robot. This project allows Spot to:
- **Use a microphone** to capture user voice commands
- **Play audio** using Spot's speaker
- **Use Spot’s camera** to capture images and process them

# Spot AI Client Architecture

## 🏗️ Overview
The Spot AI client is a modular Python project that integrates:
- **Speech recognition** for user commands.
- **Audio playback** using Spot’s speakers.
- **Camera I/O** for image capture.
- **Robot control** for executing movement commands.

## 🛠️ Components
1. `speech_recognition.py` - Processes voice commands.
2. `audio_processing.py` - Handles TTS playback and audio recording.
3. `camera_io.py` - Captures images from Spot’s onboard cameras.
4. `robot_control.py` - Executes movement commands.

## 📡 Data Flow
1. **User speaks a command** → Speech-to-text processing.
2. **Spot analyzes command** → Determines action.
3. **Action execution** → Movement, image capture, or audio playback.
4. **Spot responds** → Provides feedback via text-to-speech or image data.


## 🔧 Installation
1. Clone the repository:
   git clone https://github.com/gsurjs/SpotSDK-Cognitive-Companion.git
   cd SpotSDK-Cognitive-Companion

2. Create a Virtual Environment
    python -m venv venv
    source venv/bin/activate  # For macOS/Linux
    venv\Scripts\activate  # For Windows

3. Install dependencies:
    pip install -r requirements.txt

## 🎯 Usage
Run the Spot AI client:
    python main.py

## 🛠️ Dependencies

Python 3.6+
OpenCV
SpeechRecognition
PyAudio
bosdyn.client (Boston Dynamics SDK)

# 🔧 WIP!! 🛠️

**Thank you for your patience**