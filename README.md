# Spot: Cognitive Companion AI Client 🚀

A speech-to-speech AI client for the Boston Dynamics Spot robot. This project allows Spot to:
- **Use a microphone** to capture user voice commands
- **Play audio** using Spot's speaker
- **Use Spot’s camera** to capture images and process them

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

## 📁 Project Tree

SpotSDK-Cognitive-Companion/
│── main.py
│── README.md
│── requirements.txt
│── spot_ai/
│   ├── speech_recognition.py
│   ├── vision_processing.py
│   ├── audio_processing.py
│   ├── camera_io.py
│   ├── robot_control.py
│   ├── config.py

## 🛠️ Dependencies

Python 3.6+
OpenCV
SpeechRecognition
PyAudio
bosdyn.client (Boston Dynamics SDK)

# 🔧 WIP!! 🛠️

**Thank you for your patience**