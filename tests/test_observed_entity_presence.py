import unittest
import numpy as np
from datafarm.verify.appearance_presence import source_boundary_presence


class ObservedEntityPresenceTest(unittest.TestCase):
    def setUp(self):
        self.source = np.zeros((160, 200), np.uint8)
        self.source[50:110, 70:125] = 255
        self.empty = np.zeros_like(self.source)

    def check(self, image):
        return source_boundary_presence(self.source, self.empty, image, 100)

    def test_cool_dark_cup_has_observed_shape_and_contrast(self):
        image = np.full((160, 200, 3), 180, np.uint8)
        image[50:110, 70:125] = (35, 28, 20)
        mask, evidence = self.check(image)
        self.assertTrue(evidence[0]['supplemented'])
        self.assertEqual(np.count_nonzero(mask), np.count_nonzero(self.source))

    def test_absence_and_displaced_object_do_not_get_source_mask(self):
        for kind in ('empty', 'whole_dark', 'offset', 'broad_shadow', 'texture', 'high_contrast_texture'):
            with self.subTest(kind=kind):
                image = np.full((160, 200, 3), 180, np.uint8)
                if kind == 'whole_dark': image[:] = 30
                if kind == 'offset': image[50:110, 125:180] = 30
                if kind == 'broad_shadow': image[20:130, 30:150] = 60
                if kind == 'texture':
                    rng = np.random.default_rng(3)
                    image[:] = rng.integers(80, 120, image.shape, dtype=np.uint8)
                if kind == 'high_contrast_texture':
                    image[:] = np.random.default_rng(4).integers(0, 255, image.shape, dtype=np.uint8)
                mask, evidence = self.check(image)
                self.assertFalse(evidence[0]['supplemented'])
                self.assertFalse(mask.any())

    def test_generated_colour_mask_is_preserved(self):
        existing = self.empty.copy(); existing[5:20, 5:20] = 255
        image = np.full((160, 200, 3), 180, np.uint8)
        result, _ = source_boundary_presence(self.source, existing, image, 100)
        np.testing.assert_array_equal(result, existing)



if __name__ == '__main__': unittest.main()
