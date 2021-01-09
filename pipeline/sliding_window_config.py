from dataclasses import dataclass

@dataclass
class SlidingWindowsConfig:
    """ Properties for the Sliding Windows algorithm. """
    number_of_windows: int
    window_margin: int
    min_pixels_for_detect: int
