"""Helpers for agents: softmax and resizing of inputs / outputs."""

from __future__ import annotations

import numpy as np


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax along an axis, default the last one (float32, computed in place on a copy)."""
    result = np.array(values, dtype=np.float32, copy=True)
    result -= result.max(axis=axis, keepdims=True)
    np.exp(result, out=result)
    result /= result.sum(axis=axis, keepdims=True)
    return result


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
