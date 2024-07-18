import unittest

from nose.tools import raises

from src.utils.time import robust_hour_of_iso_date


class TestTimeUtils(unittest.TestCase):
    def test_robust_hour_of_iso_date_with_microseconds(self):
        self.assertEqual(robust_hour_of_iso_date("2015-05-12 05:25:23.904 UTC"), 5)

    def test_robust_hour_of_iso_date_without_microseconds(self):
        self.assertEqual(robust_hour_of_iso_date("2015-05-12 05:25:23 UTC"), 5)

    @raises(ValueError)
    def test_robust_hour_of_iso_date_with_invalid_iso_string(self):
        self.assertRaises(ValueError, robust_hour_of_iso_date("2015-05-12 05:25:23"))
