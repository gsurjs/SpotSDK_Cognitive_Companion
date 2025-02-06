import unittest
import os
from spot_ai.camera_io import capture_image

class TestCameraIO(unittest.TestCase):
    def test_capture_image(self):
        img_path = capture_image("frontleft_fisheye_image")
        self.assertTrue(os.path.exists(img_path))
        os.remove(img_path)  # Cleanup after test

if __name__ == "__main__":
    unittest.main()
