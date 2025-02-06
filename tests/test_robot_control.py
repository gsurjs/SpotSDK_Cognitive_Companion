import unittest
from spot_ai.robot_control import stand, sit

class TestRobotControl(unittest.TestCase):
    def test_robot_stand(self):
        response = stand()
        self.assertTrue(response)

    def test_robot_sit(self):
        response = sit()
        self.assertTrue(response)

if __name__ == "__main__":
    unittest.main()
