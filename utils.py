# By: Landon Prince (5/18/2024)

import os


# Get color based on confidence value
def get_corresponding_color(confidence):
    normalized_confidence = confidence / 100
    red = int((1 - normalized_confidence) * 255)
    green = int(normalized_confidence * 255)
    return f'\033[38;2;{red};{green};0m', (0, green, red)


# Determine if given filename has compatible extension
def is_supported(filename):
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp",
                            ".tif", ".tiff", ".webp", ".jp2"}
    extension = os.path.splitext(filename)[1].lower()
    return extension in supported_extensions
