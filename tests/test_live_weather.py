import unittest
from src.live_weather import WEATHER_CODE_LABELS, UK_CITIES


class TestLiveWeather(unittest.TestCase):
    def test_supported_cities(self):
        self.assertIn('London', UK_CITIES)
        self.assertIn('Manchester', UK_CITIES)

    def test_weather_codes(self):
        self.assertEqual(WEATHER_CODE_LABELS[0], 'Clear sky')
        self.assertEqual(WEATHER_CODE_LABELS[95], 'Thunderstorm')


if __name__ == '__main__':
    unittest.main()
