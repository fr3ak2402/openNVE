"""Tasks: how model outputs are read (classification, segmentation)."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from ..frames import Frame, Tile
from ._util import _resize, _softmax
from .config import PredictionConfig
from .registry import register_task
from .results import Prediction, lazy_output_prediction

if typing.TYPE_CHECKING:
    from .agent import Agent



@dataclasses.dataclass
class TaskInput:
    """What a task decoder gets: the agent, the frame / tiles, the input images (before
    preprocessing, height x width x channels), the padded size and the model outputs (after the
    activation, first dimension = inputs).

    logits: True if outputs[0] still holds the logits and the decoder applies the softmax itself
    (only for decoders with the attribute accepts_logits = True, for instance segmentation)."""

    agent: "Agent"
    frame: typing.Optional[Frame]
    tiles: typing.Optional[list[Tile]]
    images: list[np.ndarray]
    pad_to: typing.Optional[tuple[int, int]]
    outputs: list[np.ndarray]
    count: int
    logits: bool = False

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
    """One class per pixel: output (height, width, classes) per input, for instance a U-Net.

    Inputs whose output has the size of the input (the usual case: full tiles, no resizing) are
    decoded together with numpy operations over the whole batch; the others (smaller edge tiles,
    resized inputs) one by one.
    """
    fast = _segmentation_batch_indices(task)
    results: dict[int, Prediction] = _segmentation_batch(task, fast) if fast else {}
    slow = [index for index in range(task.count) if index not in results]
    if slow:
        single = task
        if task.logits:  # the per-input path works with probabilities
            single = dataclasses.replace(task, outputs=[task.agent._class_softmax(task.outputs[0])] + list(task.outputs[1:]), logits=False)
        results.update(_segmentation_single(single, slow))
    return [results[index] for index in range(task.count)]


_segmentation.accepts_logits = True


def _segmentation_batch_indices(task: TaskInput) -> list[int]:
    if not task.outputs or task.count == 0:
        return []
    output = task.outputs[0]
    if output.ndim != 4:
        return []
    layout = task.agent.layout
    height, width, classes = (output.shape[2], output.shape[3], output.shape[1]) if layout == "nchw" else output.shape[1:4]
    if classes < 2:
        return []
    indices = []
    for index, image in enumerate(task.images):
        if image.ndim == 3 and image.shape[:2] == (height, width):
            indices.append(index)
    return indices


def _segmentation_batch(task: TaskInput, indices: list[int]) -> dict[int, Prediction]:
    config = task.config
    output = task.outputs[0][indices]
    if task.agent.layout == "nchw":
        output = output.transpose(0, 2, 3, 1)
    values = np.asarray(output, dtype=np.float32)  # probabilities, or logits if task.logits
    classes = values.shape[3]
    class_maps = np.argmax(values, axis=3).astype(np.uint8)  # the same for logits and probabilities
    images = np.stack([task.images[index] for index in indices])
    active = np.any(images > config.activity_threshold, axis=3)
    class_counts = np.empty((len(indices), classes), np.int64)
    active_counts = np.empty((len(indices), classes), np.int64)
    for class_id in range(classes):
        is_class = class_maps == class_id
        class_counts[:, class_id] = is_class.sum(axis=(1, 2))
        active_counts[:, class_id] = (is_class & active).sum(axis=(1, 2))
    targets = config.target_classes if config.target_classes is not None else list(range(1, classes))
    target_mask = np.isin(class_maps, targets)
    if config.mask_inactive:
        target_mask &= active
    if config.threshold is not None:
        if task.logits:  # probability of the most likely class: 1 / sum(exp(logit - max logit))
            best = 1.0 / np.exp(values - values.max(axis=3, keepdims=True)).sum(axis=3)
        else:
            best = np.take_along_axis(values, class_maps[..., None].astype(np.intp), axis=3)[..., 0]
        target_mask &= best >= config.threshold
    batch_index, ys, xs = np.nonzero(target_mask)
    pixel_classes = class_maps[batch_index, ys, xs]
    all_scores = values[batch_index, ys, xs]
    if task.logits:
        all_scores = _softmax(all_scores)
    bounds = np.searchsorted(batch_index, np.arange(len(indices) + 1))
    results = {}
    for position, index in enumerate(indices):
        first, last = bounds[position], bounds[position + 1]
        tile = task.tile(index)
        offset_x, offset_y = (tile.x, tile.y) if tile is not None else (0, 0)
        pixels = np.zeros(last - first, _PIXEL_DTYPE)
        pixels["x"] = xs[first:last] + offset_x
        pixels["y"] = ys[first:last] + offset_y
        pixels["class_id"] = pixel_classes[first:last]
        pixel_scores = all_scores[first:last]
        if last > first:
            pixels["score"] = pixel_scores[np.arange(last - first), pixels["class_id"]]
        results[index] = _segmentation_prediction(
            task,
            index,
            tile,
            values[position],
            class_maps[position],
            pixels,
            pixel_scores,
            class_counts[position],
            np.stack([class_counts[position] - active_counts[position], active_counts[position]]),
        )
    return results


def _segmentation_prediction(task, index, tile, values, class_map, pixels, pixel_scores, class_counts, activity_counts):
    """values: probabilities (height x width x classes), or logits if task.logits."""
    config = task.config
    class_id = label = score = None
    if len(pixels):
        class_id = int(np.bincount(pixels["class_id"]).argmax())
        score = float(pixels["score"][pixels["class_id"] == class_id].mean())
        if config.labels and class_id < len(config.labels):
            label = config.labels[class_id]
    fields = dict(
        tile=tile,
        outputs=[output[index] for output in task.outputs],
        class_id=class_id,
        label=label,
        score=score,
        selected=len(pixels) >= max(config.min_pixels, 1),
        class_map=class_map,
        pixels=pixels,
        pixel_scores=pixel_scores,
        class_counts=class_counts,
        activity_counts=activity_counts,
        labels=config.labels,
    )
    if task.logits:
        rest = fields.pop("outputs")[1:]
        return lazy_output_prediction(values, _softmax, rest, **fields)
    return Prediction(output=values, **fields)


def _segmentation_single(task: TaskInput, indices: list[int]) -> dict[int, Prediction]:
    config = task.config
    layout = task.agent.layout
    predictions = {}
    for index in indices:
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

        predictions[index] = _segmentation_prediction(
            task, index, tile, probabilities, class_map, pixels, pixel_scores, class_counts, activity_counts
        )
    return predictions
