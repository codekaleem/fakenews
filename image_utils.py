import io
import math
from typing import Dict

import numpy as np
from PIL import Image, ImageStat


def compute_entropy(image: Image.Image) -> float:
    histogram = image.convert('L').histogram()
    histogram_size = sum(histogram)
    if histogram_size == 0:
        return 0.0
    entropy = -sum(
        (count / histogram_size) * math.log2(count / histogram_size)
        for count in histogram
        if count
    )
    return float(entropy)


def compute_color_stats(image: Image.Image) -> Dict[str, float]:
    array = np.array(image.convert('RGB'))
    if array.size == 0:
        return {'unique_colors': 0, 'stddev': 0.0}

    pixels = array.reshape(-1, 3)
    unique_colors = int(np.unique(pixels, axis=0).shape[0])
    stddev = float(np.std(pixels))
    return {'unique_colors': unique_colors, 'stddev': stddev}


def image_suspicion_score(image: Image.Image) -> float:
    analysis_image = image.copy()
    analysis_image.thumbnail((512, 512))

    entropy = compute_entropy(analysis_image)
    stats = compute_color_stats(analysis_image)
    unique_colors = stats['unique_colors']
    stddev = stats['stddev']
    width, height = analysis_image.size
    aspect_ratio = max(width / height, height / width)

    score = 0.2
    if entropy < 4.0:
        score += 0.25
    elif entropy < 5.0:
        score += 0.1

    if unique_colors < 256:
        score += 0.2
    elif unique_colors < 1024:
        score += 0.1

    if stddev < 40.0:
        score += 0.2
    elif stddev < 60.0:
        score += 0.1

    if aspect_ratio > 2.5:
        score += 0.1

    return min(score, 1.0)


def analyze_image(image_bytes: bytes) -> Dict[str, object]:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image.thumbnail((1024, 1024))
    width, height = image.size
    stats = compute_color_stats(image)
    entropy = compute_entropy(image)
    suspicion_score = image_suspicion_score(image)

    return {
        'format': image.format or 'UNKNOWN',
        'width': width,
        'height': height,
        'mode': image.mode,
        'size_bytes': len(image_bytes),
        'entropy': round(entropy, 2),
        'unique_colors': stats['unique_colors'],
        'stddev': round(stats['stddev'], 2),
        'suspicion_score': round(suspicion_score, 2),
    }
