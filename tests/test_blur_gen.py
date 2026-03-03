import unittest
import numpy as np
import cv2
import os
import sys

# Add project root to path so we can import project.blur_augment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project.blur_augment import BlurAugment

class TestBlurAugment(unittest.TestCase):
    def setUp(self):
        self.aug = BlurAugment(blur_prob=1.0, min_kernel=5, max_kernel=9)
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some white pixels to check blurring
        cv2.rectangle(self.image, (40, 40), (60, 60), (255, 255, 255), -1)

    def test_output_shape(self):
        """Test that the output shape matches the input shape."""
        blurred = self.aug(self.image)
        self.assertEqual(self.image.shape, blurred.shape)

    def test_blur_applied(self):
        """Test that the image is actually modified (blurred)."""
        blurred = self.aug(self.image)
        # The blurred image should be different from the original
        diff = np.sum(np.abs(self.image.astype(int) - blurred.astype(int)))
        self.assertGreater(diff, 0, "Blurred image is identical to original!")

    def test_probabilistic_bypass(self):
        """Test that blur_prob=0 returns original image."""
        no_aug = BlurAugment(blur_prob=0.0)
        out = no_aug(self.image)
        np.testing.assert_array_equal(out, self.image)

if __name__ == '__main__':
    unittest.main()
