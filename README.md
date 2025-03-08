# Spot Robot Speech-To-Speech AI Client

This repository contains a Python client that integrates the Boston Dynamics Spot robot with speech recognition and AI language models to create an interactive voice assistant. The system uses Groq and Google Gemini for AI processing, Whisper for speech recognition, and the Boston Dynamics SDK to interface with Spot's hardware.

## Features

- **Wake Word Detection**: Activates when you say "spot" followed by your command
- **Multi-Modal Input**: Can use Spot's cameras to provide visual context to the AI
- **Multiple Camera Support**: Front fisheye, rear fisheye, and PTZ camera (if Spot CAM is available)
- **Audio I/O**: Uses Spot's microphones for input and speakers for output
- **AI-Powered Responses**: Uses Groq's Llama 3 70B model for high-quality responses
- **Visual Understanding**: Uses Google Gemini for image analysis when visual context is needed
- **Function Calling**: Intelligently decides which of Spot's sensors to use based on your query

## Prerequisites

- Boston Dynamics Spot robot with SDK access
- Python 3.8 or higher
- API keys for Groq and Google Gemini
- Network access to the Spot robot

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/spot-ai-client.git
cd spot-ai-client
```

### 2. Set Up a Virtual Environment

#### On Windows:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Edit the `spot_ai_client.py` file to include your API keys:

```python
groq_client = Groq(api_key="YOUR_GROQ_API_KEY")
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

## Usage

### Basic Usage

There are two ways to authenticate with the robot:

#### 1. Using command-line credentials:

```bash
python spot_ai_client.py --hostname 192.168.80.3 --username your_username --password your_password
```

#### 2. Using the .bosdyn credential store (recommended for development):

If you've previously authenticated with the robot using the Boston Dynamics SDK tools, you can use:

```bash
python spot_ai_client.py --hostname 192.168.80.3
```

This method relies on credentials stored in the `.bosdyn` folder in your home directory.

### Specify a Different IP Address

If your Spot robot is at a different IP address:

```bash
python spot_ai_client.py --hostname 192.168.X.X --username your_username --password your_password
```

### Interacting with Spot

1. After starting the program, the robot will listen for the wake word "spot"
2. When you say "spot" followed by your command (e.g., "spot what can you see?"), the AI will process your request
3. Depending on your command, Spot might:
   - Capture an image from one of its cameras
   - Process clipboard content
   - Simply respond with AI-generated text
4. The response will be spoken through Spot's audio system

## Configuration

### Camera Sources

The system supports multiple camera configurations:

- `front camera` - Uses the front fisheye camera
- `rear camera` - Uses the rear fisheye camera
- `ptz camera` - Uses the PTZ camera (if Spot CAM is available)

### Audio Settings

The default audio configuration uses:

- 16000Hz sample rate
- Single channel audio
- 16-bit audio depth

These settings can be adjusted in the `SpotAIClient` class initialization.

## Authentication

Boston Dynamics Spot robots require authentication to access the API. There are two main methods:

### 1. Username and Password

You can provide credentials directly when running the script:

```bash
python spot_ai_client.py --hostname 192.168.80.3 --username your_username --password your_password
```

**Note**: Providing passwords on the command line is not secure for production environments. Consider using environment variables or the credential store method.

### 2. Credential Store (.bosdyn folder)

The Boston Dynamics SDK provides a credential store mechanism that securely saves your authentication information:

1. First, use the `bosdyn` utility to store your credentials:
   ```bash
   python -m bosdyn.client 192.168.80.3 your_username your_password
   ```

2. Then run the client without explicit credentials:
   ```bash
   python spot_ai_client.py --hostname 192.168.80.3
   ```

This stores your credentials in the `~/.bosdyn` directory and is the recommended approach for development.

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Make sure you're using the correct username and password
   - Verify your account has the necessary permissions on the robot
   - Try using the SDK's authentication utility to test your credentials

2. **Connection Errors**:
   - Ensure you can ping the Spot robot from your computer
   - Verify you have the correct IP address
   - Check that your account has API access to the robot

2. **Audio Problems**:
   - If Spot's speakers aren't working, the system will fall back to local TTS
   - Check the robot's volume settings using the Spot tablet app

3. **Camera Errors**:
   - Different Spot configurations have different cameras available
   - If the requested camera is not available, the system will log an error

4. **API Key Issues**:
   - Verify your Groq and Gemini API keys are correctly entered
   - Check your API usage quotas if requests are failing

## Additional Resources

- [Boston Dynamics Spot SDK Documentation](https://dev.bostondynamics.com/)
- [Groq AI Documentation](https://console.groq.com/docs)
- [Google Gemini Documentation](https://ai.google.dev/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original code by Robert Stanley
- Modified for Boston Dynamics Spot SDK integration