from setuptools import setup, find_packages

setup(
    name="spot_ai_client",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "bosdyn-client",
        "bosdyn-api",
        "bosdyn-mission",
        "bosdyn-choreography-client",
        "bosdyn-orbit",
        "openai",
        "faster-whisper",
        "speechrecognition",
        "pyaudio",
        "pyttsx3",
        "opencv-python",
        "google-generativeai",
        "pillow",
        "pyperclip",
    ],
    entry_points={
        "console_scripts": [
            "spot_ai=spot_ai_client.main:main"
        ]
    },
)
