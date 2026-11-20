"""Tasks: how model outputs are read (classification, segmentation)."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from ..frames import Frame, Tile
from ._util import _resize
from .config import PredictionConfig
from .registry import register_task
from .results import Prediction

if typing.TYPE_CHECKING:
    from .agent import Agent



@dataclasses.dataclass
class TaskInput:
    """What a task decoder gets: the agent, the frame / tiles, the input images (before
    preprocessing, height x width x channels), the padded size and the model outputs (after the
    activation, first dimension = inputs)."""

    agent: "Agent"
    frame: typing.Optional[Frame]
    tiles: typing.Optional[list[Tile]]
    images: list[np.ndarray]
    pad_to: typing.Optional[tuple[int, int]]
    outputs: list[np.ndarray]
    count: int

    @property
    def config(self) -> PredictionConfig:
        return self.agent.config.prediction

    def tile(self, index: int) -> typing.Optional[Tile]:
        return None if self.tiles is None else self.tiles[index]


@register_task("classification")
def _classification(task: TaskInput) -> list[Prediction]:
    prediction_config = task.config
    outputs = task.outputs
    predictions = []
    for index in range(task.count):
        per_input = [output[index] for output in outputs]
        output = per_input[0] if per_input else np.zeros(0)
        class_id = label = score = None
        values = np.asarray(output).reshape(-1) if np.asarray(output).ndim <= 1 else None
        if values is not None and values.size > 0:
            if values.size == 1:
                score = float(values[0])
                class_id = int(score >= 0.5)
            else:
                class_id = int(np.argmax(values))
                score = float(values[class_id])
            labels = prediction_config.labels
            if labels and class_id is not None and class_id < len(labels):
                label = labels[class_id]
        selected = (
            True
            if prediction_config.threshold is None
            else (score is not None and score >= prediction_config.threshold)
        )
        predictions.append(
            Prediction(
                tile=task.tile(index),
                output=np.asarray(output),
                outputs=per_input,
                class_id=class_id,
                label=label,
                score=score,
                selected=selected,
                labels=prediction_config.labels,
            )
        )
    return predictions


_PIXEL_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("class_id", "u1"), ("score", "<f4")])


@register_task("segmentation")
def _segmentation(task: TaskInput) -> list[Prediction]:
    """One class per pixel: output (height, width, classes) per input, for instance a U-Net."""
    config = task.config
    layout = task.agent.config.preprocess.layout
    predictions = []
    for index in range(task.count):
        per_input = [output[index] for output in task.outputs]
        probabilities = np.asarray(per_input[0], dtype=np.float32)
        if probabilities.ndim == 2:  # one class map channel
            probabilities = probabilities[:, :, None]
        if probabilities.ndim != 3:
            raise ValueError(
                f"agent {task.agent.name}: segmentation needs outputs of height x width x classes, "
                f"got {probabilities.shape}"
            )
        if layout == "nchw":
            probabilities = probabilities.transpose(1, 2, 0)
        image = task.images[index]
        if image.ndim == 2:
            image = image[:, :, None]
        height, width = image.shape[:2]
        # output -> input size (padded tiles: padded size), then crop to the tile
        source_height, source_width = task.pad_to if task.pad_to is not None else (height, width)
        if probabilities.shape[:2] != (source_height, source_width):
            probabilities = _resize(probabilities, source_width, source_height)
        probabilities = probabilities[:height, :width]
        classes = probabilities.shape[2]
        if classes == 1:  # binary segmentation with one sigmoid channel
            probabilities = np.concatenate([1 - probabilities, probabilities], axis=2)
            classes = 2
        class_map = np.argmax(probabilities, axis=2).astype(np.uint8)
        active = np.any(image > config.activity_threshold, axis=2)
        class_counts = np.bincount(class_map.ravel(), minlength=classes).astype(np.int64)
        activity_counts = np.stack(
            [
                np.bincount(class_map[~active], minlength=classes),
                np.bincount(class_map[active], minlength=classes),
            ]
        ).astype(np.int64)

        targets = config.target_classes if config.target_classes is not None else list(range(1, classes))
        target_mask = np.isin(class_map, targets)
        if config.mask_inactive:
            target_mask &= active
        if config.threshold is not None:
            target_mask &= np.take_along_axis(probabilities, class_map[:, :, None].astype(np.intp), axis=2)[:, :, 0] >= config.threshold
        ys, xs = np.nonzero(target_mask)
        tile = task.tile(index)
        offset_x, offset_y = (tile.x, tile.y) if tile is not None else (0, 0)
        pixels = np.zeros(len(ys), _PIXEL_DTYPE)
        pixels["x"] = xs + offset_x
        pixels["y"] = ys + offset_y
        pixels["class_id"] = class_map[ys, xs]
        pixel_scores = probabilities[ys, xs]
        pixels["score"] = pixel_scores[np.arange(len(ys)), pixels["class_id"]] if len(ys) else 0

        class_id = label = score = None
        if len(ys):
            class_id = int(np.bincount(pixels["class_id"]).argmax())
            score = float(pixels["score"][pixels["class_id"] == class_id].mean())
            if config.labels and class_id < len(config.labels):
                label = config.labels[class_id]
        predictions.append(
            Prediction(
                tile=tile,
                output=probabilities,
                outputs=per_input,
                class_id=class_id,
                label=label,
                score=score,
                selected=len(ys) >= max(config.min_pixels, 1),
                class_map=class_map,
                pixels=pixels,
                pixel_scores=pixel_scores,
                class_counts=class_counts,
                activity_counts=activity_counts,
                labels=config.labels,
            )
        )
    return predictions
