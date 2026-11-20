"""Helpers for agents: softmax and resizing of inputs / outputs."""

from __future__ import annotations

import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    import cv2

    channels = image.shape[2]
    if channels <= 4:
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    else:
        resized = np.stack(
            [cv2.resize(image[:, :, c], (width, height), interpolation=cv2.INTER_AREA) for c in range(channels)],
            axis=2,
        )
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return resized
