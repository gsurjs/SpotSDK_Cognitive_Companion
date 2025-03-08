"""
Author: Robert Stanley (Modified for Boston Dynamics Spot SDK)
Spot Robot Speech-To-Speech AI client
Free Version without OpenAI TTS paid token model
Uses Boston Dynamics Spot SDK for robot I/O operations
"""

from groq import Groq
from openai import OpenAI
from faster_whisper import WhisperModel
import google.generativeai as genai
import os
import time
import re
import pyttsx3
import io
import numpy as np
from PIL import Image
import argparse
import pyperclip
import tempfile
import wave

# Spot SDK imports
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot import Robot
from bosdyn.client.image import ImageClient
from bosdyn.client.microphone import MicrophoneClient
from bosdyn.api import image_pb2
from bosdyn.client.audio import AudioClient
from bosdyn.api import audio_pb2
from bosdyn.client import spot_cam
from bosdyn.client.spot_cam.audio import AudioClient as SpotCamAudioClient

# Wake word for SpotAI
wake_word = 'spot'
groq_client = Groq(api_key="YOUR API KEY HERE FOR GROQ OLLAMA")
genai.configure(api_key="YOUR API KEY HERE FOR GEMINI")
# openai_client = OpenAI(api_key="YOUR API KEY HERE FOR OPENAI TTS CLIENT")

# LLama sys message prompt
sys_msg = (
    'You are a multi-modal AI voice assistant powering a Boston Dynamics Spot robot. Your user may or may not have attached a photo for context '
    '(either a screenshot or a robot camera capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]

# Google Gemini prompt settings
generation_config = {
    'temperature': 0.7, 
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

# Turn off all Gemini safety settings
safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT', 
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

# Gemini model call
model = genai.GenerativeModel('gemini-1.5-flash-latest',
                            generation_config=generation_config,
                            safety_settings=safety_settings)

# Check and store num cpu cores using os library
num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)

class SpotAIClient:
    def __init__(self, hostname, username=None, password=None):
        # Initialize robot connection
        self.robot = self._setup_robot(hostname, username, password)
        
        # Initialize Spot SDK clients
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.microphone_client = self.robot.ensure_client(MicrophoneClient.default_service_name)
        self.audio_client = self.robot.ensure_client(AudioClient.default_service_name)
        
        # Check if Spot CAM is available and set up its clients
        try:
            self.spot_cam_client = self.robot.ensure_client('spot-cam-api')
            self.spot_cam_audio = spot_cam.AudioClient(self.spot_cam_client)
            self.has_spot_cam = True
            print("Spot CAM detected and initialized")
        except:
            self.has_spot_cam = False
            print("Spot CAM not detected, using onboard systems only")
        
        # Set up audio parameters
        self.audio_sample_rate = 16000  # Hz
        self.audio_channels = 1
        
        # Initialize temporary directory for files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Initialize listening state
        self.is_listening = False
        self.last_audio_time = time.time()
        self.buffer_duration = 0.5  # seconds between audio checks
        
    def _setup_robot(self, hostname, username=None, password=None):
        """Set up the robot client and authenticate"""
        # Create robot object
        sdk = bosdyn.client.create_standard_sdk("SpotAI Client")
        robot = sdk.create_robot(hostname)
        
        # Authenticate with the robot using provided credentials
        if username and password:
            robot.authenticate(username, password)
        else:
            # Use the authentication from the robot's .bosdyn folder
            bosdyn.client.util.authenticate(robot)
        
        # Establish time sync with the robot
        robot.time_sync.wait_for_sync()
        
        # Acquire robot lease
        robot.logger.info("Acquiring lease...")
        lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        lease = lease_client.acquire()
        
        return robot
        
    def groq_prompt(self, prompt, img_context):
        """Send a prompt to the Groq LLM API with optional image context"""
        if img_context:
            prompt = f'USER PROMPT: {prompt}\n\n    IMAGE CONTEXT: {img_context}'
        convo.append({'role': 'user', 'content': prompt})
        chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
        response = chat_completion.choices[0].message
        convo.append(response)
        
        return response.content

    def function_call(self, prompt):
        """Determine which functions to call based on the user prompt"""
        sys_msg = (
            'You are an AI function calling model for a Boston Dynamics Spot robot. You will determine whether '
            'extracting clipboard content (if available), taking a camera image, or calling no functions is best '
            'for the robot to respond to the user prompt. You will respond with only a selection from this list: '
            '["extract clipboard", "capture front camera", "capture rear camera", "capture ptz camera", "None"] \n'
            'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
            'function call name exactly as listed.'
        )

        function_convo = [{'role': 'system', 'content': sys_msg},
                        {'role': 'user', 'content': prompt}]
        
        chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
        response = chat_completion.choices[0].message

        return response.content

    def capture_camera_image(self, camera_source="frontleft_fisheye_image"):
        """Capture an image from one of Spot's cameras"""
        # Map of camera sources
        camera_sources = {
            "front camera": "frontleft_fisheye_image",
            "rear camera": "back_fisheye_image",
            "ptz camera": "ptz"  # For Spot CAM
        }
        
        # Get the actual camera source string if a friendly name was used
        if camera_source in camera_sources:
            camera_source = camera_sources[camera_source]
        
        # Handle Spot CAM PTZ camera
        if camera_source == "ptz" and self.has_spot_cam:
            # Use the Spot CAM API to get the PTZ camera image
            ptz_client = self.robot.ensure_client(spot_cam.PtzClient.default_service_name)
            image_request = spot_cam.GetImageRequest(camera=spot_cam.Camera.PTZ)
            image_response = ptz_client.get_image(image_request)
            image_data = Image.open(io.BytesIO(image_response.image))
            
        else:
            # Use the standard Spot cameras
            image_request = [image_pb2.ImageRequest(image_source_name=camera_source, 
                                                  quality_percent=70,
                                                  image_format=image_pb2.Image.FORMAT_JPEG)]
            image_response = self.image_client.get_image(image_request)
            
            if not image_response:
                print(f"No image data received from {camera_source}")
                return None
                
            # Convert the image data to a PIL Image
            image_data = Image.open(io.BytesIO(image_response[0].shot.image.data))
        
        # Save the image to a temporary file
        image_path = os.path.join(self.temp_dir.name, "spot_camera.jpg")
        image_data.save(image_path, quality=70)
        return image_path

    def get_clipboard_text(self):
        """Get text from clipboard of the controlling computer"""
        # Note: This still uses the client computer's clipboard as Spot doesn't have one
        clipboard_content = pyperclip.paste()
        if isinstance(clipboard_content, str):
            return clipboard_content
        else:
            print("No clipboard text to copy")
            return None

    def vision_prompt(self, prompt, photo_path):
        """Use Gemini to analyze an image and provide context"""
        img = Image.open(photo_path) 
        prompt = (
            'You are the vision analysis AI that provides semantic meaning from images to provide context '
            'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
            'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
            'relevant to the user prompt. Then generate as much objective data about the image for the AI '
            f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
        )

        response = model.generate_content([prompt, img])
        return response.text

    def speak(self, text):
        """Convert text to speech and play it on Spot's speaker"""
        # Option 1: Use pyttsx3 to generate speech on the client computer (fallback)
        def speak_with_pyttsx3(text):
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            
        # Option 2: Use Spot's built-in text-to-speech capabilities
        def speak_with_spot(text):
            try:
                # Use Spot's audio service to play the text
                audio_request = audio_pb2.TextToSpeechRequest(text=text)
                response = self.audio_client.text_to_speech(audio_request)
                return True
            except Exception as e:
                print(f"Error using Spot's text-to-speech: {e}")
                return False
                
        # Option 3: Use Spot CAM's audio system if available
        def speak_with_spot_cam(text):
            if not self.has_spot_cam:
                return False
                
            try:
                # Use Spot CAM's text-to-speech capabilities
                self.spot_cam_audio.play_sound(text=text)
                return True
            except Exception as e:
                print(f"Error using Spot CAM's text-to-speech: {e}")
                return False
        
        # Try each option in order of preference
        if not speak_with_spot(text):
            if not speak_with_spot_cam(text):
                speak_with_pyttsx3(text)

    def record_audio(self, duration=5):
        """Record audio from Spot's microphones"""
        # Set up the audio recording request
        request = bosdyn.client.microphone.MicrophoneKeepAliveRequest(
            frequency=10.0,  # How often to send KeepAlive messages
            audio_type=bosdyn.client.microphone.AudioType.AudioType.RAW_CAP
        )
        
        # Start recording
        print("Listening...")
        keep_alive = self.microphone_client.stream_microphone_data(request)
        
        # Collect audio data for duration seconds
        start_time = time.time()
        audio_data = bytearray()
        
        try:
            for audio_data_response in keep_alive:
                audio_data.extend(audio_data_response.data)
                if time.time() - start_time > duration:
                    break
        except Exception as e:
            print(f"Error recording audio: {e}")
        finally:
            keep_alive.cancel()
        
        # Save the audio data to a WAV file
        audio_path = os.path.join(self.temp_dir.name, "spot_audio.wav")
        
        with wave.open(audio_path, 'wb') as wav_file:
            wav_file.setnchannels(self.audio_channels)
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(self.audio_sample_rate)
            wav_file.writeframes(audio_data)
        
        return audio_path

    def wav_to_text(self, audio_path):
        """Convert WAV audio to text using Whisper"""
        segments, _ = whisper_model.transcribe(audio_path)
        text = ''.join(segment.text for segment in segments)
        return text

    def extract_prompt(self, transcribed_text):
        """Extract the prompt after the wake word"""
        pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)' # regex search for wake word and the text following it
        match = re.search(pattern, transcribed_text, re.IGNORECASE)
        
        if match:
            prompt = match.group(1).strip() # select and strip words after wake word
            return prompt
        else:
            return None

    def listen_for_commands(self):
        """Main function to listen for commands in a loop"""
        print(f"Listening for the wake word '{wake_word}'...")
        
        while True:
            # Record a short audio clip
            audio_path = self.record_audio(duration=5)
            
            # Convert audio to text
            transcribed_text = self.wav_to_text(audio_path)
            
            # Check for wake word
            clean_prompt = self.extract_prompt(transcribed_text)
            
            if clean_prompt:
                print(f"\nUSER: {clean_prompt}\n")
                
                # Determine what function to call
                call = self.function_call(clean_prompt)
                
                # Execute the appropriate function
                if 'capture front camera' in call:
                    print('Capturing front camera image...')
                    photo_path = self.capture_camera_image("front camera")
                    visual_context = self.vision_prompt(prompt=clean_prompt, photo_path=photo_path)
                elif 'capture rear camera' in call:
                    print('Capturing rear camera image...')
                    photo_path = self.capture_camera_image("rear camera")
                    visual_context = self.vision_prompt(prompt=clean_prompt, photo_path=photo_path)
                elif 'capture ptz camera' in call:
                    print('Capturing PTZ camera image...')
                    photo_path = self.capture_camera_image("ptz camera")
                    visual_context = self.vision_prompt(prompt=clean_prompt, photo_path=photo_path)
                elif 'extract clipboard' in call:
                    print('Extracting clipboard text.')
                    paste = self.get_clipboard_text()
                    clean_prompt = f'{clean_prompt} \n\n    CLIPBOARD CONTENT: {paste}'
                    visual_context = None
                else:
                    visual_context = None

                # Get response from LLM
                response = self.groq_prompt(prompt=clean_prompt, img_context=visual_context)
                print(f'\nSPOT AI: {response}')
                
                # Speak the response
                self.speak(response)
                
            # Pause before listening again
            time.sleep(0.5)

    def cleanup(self):
        """Clean up resources when done"""
        # Clean up temporary directory
        self.temp_dir.cleanup()
        
        # Return lease on the robot
        try:
            self.robot.return_lease()
        except:
            pass

def main():
    """Main function to parse arguments and start the Spot AI client"""
    parser = argparse.ArgumentParser(description='Spot Robot Speech-To-Speech AI client')
    parser.add_argument('--hostname', default='192.168.80.3', help='Hostname or IP address of the robot')
    parser.add_argument('--username', help='Username for robot authentication')
    parser.add_argument('--password', help='Password for robot authentication')
    args = parser.parse_args()
    
    try:
        spot_client = SpotAIClient(args.hostname, args.username, args.password)
        spot_client.listen_for_commands()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'spot_client' in locals():
            spot_client.cleanup()

if __name__ == "__main__":
    main()