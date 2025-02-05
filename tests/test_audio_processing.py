import unittest
from spot_ai.audio_processing import play_audio_on_spot

class TestAudioProcessing(unittest.TestCase):
    def test_audio_playback(self):
        self.assertIsNone(play_audio_on_spot("test_audio.wav"))

if __name__ == "__main__":
    unittest.main()
