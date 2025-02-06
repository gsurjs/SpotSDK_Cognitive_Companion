import unittest
from spot_ai.speech_recognition import transcribe_audio

class TestSpeechRecognition(unittest.TestCase):
    def test_transcription(self):
        transcript = transcribe_audio("tests/sample_voice_command.wav")
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0)

if __name__ == "__main__":
    unittest.main()
