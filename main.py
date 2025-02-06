"""
Author: Robert Stanley
Spot Robot Speech-To-Speech AI client
Modified to include Spot's Microphone, Speaker, and Camera access
"""

from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.audio import AudioClient
from bosdyn.client.spot_cam.audio import AudioCapture
from bosdyn.api.spot_cam import audio_pb2
from bosdyn.api import data_chunk_pb2
from groq import Groq
from PIL import ImageGrab, Image
import speech_recognition as sr
import cv2
import pyaudio
import wave
import os
import time
import re
import pyttsx3
import numpy as np

# Set up Spot SDK
SDK = create_standard_sdk("Spot-Mic-Speaker-Camera")
robot = SDK.create_robot("192.168.80.3")  # Replace with Spot's actual IP address
robot.authenticate("user", "password")  # Replace with Spot's credentials

# Create service clients
audio_client = robot.ensure_client(AudioClient.default_service_name)
image_client = robot.ensure_client(ImageClient.default_service_name)
audio_capture_client = robot.ensure_client(AudioCapture.default_service_name)

# Wake word for SpotAI
wake_word = "spot"
groq_client = Groq(api_key="YOUR API KEY HERE FOR GROQ OLLAMA")

r = sr.Recognizer()
source = sr.Microphone()


def capture_image(source_name="frontleft_fisheye_image"):
    """Capture an image from Spot's camera."""
    image_responses = image_client.get_image_from_sources([source_name])
    image = image_responses[0].shot.image

    # Save image
    image_path = f"spot_{source_name}.jpg"
    with open(image_path, "wb") as img_file:
        img_file.write(image.data)

    print(f"Image saved: {image_path}")
    return image_path


def capture_audio_from_spot(duration=5):
    """Capture audio from Spot's microphone."""
    print(f"Recording audio from Spot's microphone for {duration} seconds...")
    audio_capture = audio_capture_client.capture_audio(duration_sec=duration)
    audio_data = b"".join(chunk.data for chunk in audio_capture.chunks)

    # Save as WAV file
    audio_path = "spot_audio.wav"
    with wave.open(audio_path, "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(16000)
        wave_file.writeframes(audio_data)

    print(f"Audio saved: {audio_path}")
    return audio_path


def play_audio_on_spot(audio_file="spot_audio.wav"):
    """Play audio on Spot's speaker."""
    print(f"Playing {audio_file} on Spot's speaker...")
    
    # Read audio file
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    
    # Convert to data chunk format
    data_chunk = data_chunk_pb2.DataChunk(data=audio_data)
    audio_stream = audio_pb2.PlayAudioRequest(
        audio=data_chunk,
        sample_rate=16000,
        channels=1,
        sample_format=audio_pb2.PCM_S16LE,
    )

    # Play audio
    audio_client.play_audio(audio_stream)
    print("Audio playback complete.")


def callback(recognizer, audio):
    """Process voice command and interact with Spot."""
    prompt_audio_path = "prompt.wav"
    with open(prompt_audio_path, "wb") as f:
        f.write(audio.get_wav_data())

    # Transcribe audio
    with sr.AudioFile(prompt_audio_path) as source:
        audio_text = r.record(source)
        try:
            command = r.recognize_google(audio_text)
            print(f"User said: {command}")

            # Execute actions
            if "take picture" in command:
                img_path = capture_image()
                print(f"Image captured: {img_path}")

            elif "record audio" in command:
                audio_path = capture_audio_from_spot()
                print(f"Audio recorded: {audio_path}")

            elif "play audio" in command:
                play_audio_on_spot()
                print("Audio playback triggered.")

            else:
                print("Unrecognized command.")

        except sr.UnknownValueError:
            print("Could not understand the command.")


def start_listening():
    """Start listening for commands."""
    with source as s:
        r.adjust_for_ambient_noise(s, duration=0.5)

    print("\nSay 'Spot' followed by a command.")
    r.listen_in_background(source, callback)

    while True:
        time.sleep(0.5)


if __name__ == "__main__":
    start_listening()
