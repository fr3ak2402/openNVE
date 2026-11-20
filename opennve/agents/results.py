"""Results of agents: Prediction (per tile or frame) and AgentResult (per frame)."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from ..frames import Frame, Tile



@dataclasses.dataclass
class Prediction:
    """Prediction of an agent for one tile (or for the whole frame).

    Attributes:
        tile: The tile (None for "input": "frame").
        output: Output of the model for this input (first model output, after the activation).
        outputs: All model outputs for this input.
        class_id: Index of the highest value if the output is a vector (else None).
        label: prediction.labels[class_id] if labels are configured (else None).
        score: Highest value of the output vector (one value: that value), else None.
        selected: score >= threshold (True if no threshold is configured).
    """

    tile: typing.Optional[Tile]
    output: np.ndarray
    outputs: list[np.ndarray]
    class_id: typing.Optional[int] = None
    label: typing.Optional[str] = None
    score: typing.Optional[float] = None
    selected: bool = True
    class_map: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    pixels: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    pixel_scores: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    class_counts: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    activity_counts: typing.Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    labels: typing.Optional[list[str]] = dataclasses.field(default=None, repr=False)

    @property
    def tile_id(self) -> typing.Optional[int]:
        return None if self.tile is None else self.tile.id

    @property
    def pixel_count(self) -> int:
        """Segmentation: number of target pixels."""
        return 0 if self.pixels is None else len(self.pixels)

    def class_name(self, class_id: int) -> str:
        if self.labels and class_id < len(self.labels):
            return self.labels[class_id]
        return f"class_{class_id}"

    def pixel_list(self) -> list[dict[str, typing.Any]]:
        """Segmentation: target pixels as dictionaries {x, y, class_id, score, scores: {class name: probability}}
        (x, y in frame coordinates)."""
        if self.pixels is None:
            return []
        names = [self.class_name(index) for index in range(self.pixel_scores.shape[1])] if len(self.pixels) else []
        return [
            {
                "x": int(pixel["x"]),
                "y": int(pixel["y"]),
                "class_id": int(pixel["class_id"]),
                "score": float(pixel["score"]),
                "scores": {name: float(value) for name, value in zip(names, scores)},
            }
            for pixel, scores in zip(self.pixels, self.pixel_scores)
        ]

    def to_dict(self) -> dict[str, typing.Any]:
        """JSON-friendly summary (without the tile data)."""
        result: dict[str, typing.Any] = {}
        if self.tile is not None:
            result.update(
                {
                    "tile_id": self.tile.id,
                    "row": self.tile.row,
                    "col": self.tile.col,
                    "x": self.tile.x,
                    "y": self.tile.y,
                    "width": self.tile.width,
                    "height": self.tile.height,
                }
            )
        result.update({"class_id": self.class_id, "label": self.label, "score": self.score})
        if self.class_map is None:
            result["output"] = self.output.tolist()
        else:
            result["class_counts"] = {
                self.class_name(index): int(count) for index, count in enumerate(self.class_counts)
            }
            result["pixels"] = self.pixel_list()
        return result


@dataclasses.dataclass
class AgentResult:
    """Result of one agent for one frame.

    Attributes:
        agent: Agent name.
        frame: The frame (None if the agent was called with tiles or an array).
        predictions: One prediction per tile (or one for the whole frame), in tile order.
        outputs: Raw model outputs for all inputs (first dimension = number of inputs).
        inference_ms: Time spent in the model (without preprocessing).
    """

    agent: str
    frame: typing.Optional[Frame]
    predictions: list[Prediction]
    outputs: list[np.ndarray]
    inference_ms: float = 0.0

    @property
    def selected(self) -> list[Prediction]:
        """Predictions with score >= threshold (all predictions if no threshold is set)."""
        return [prediction for prediction in self.predictions if prediction.selected]

    def __len__(self) -> int:
        return len(self.predictions)

    def __iter__(self):
        return iter(self.predictions)

    def by_tile(self) -> dict[int, Prediction]:
        """{tile id: prediction}."""
        return {prediction.tile.id: prediction for prediction in self.predictions if prediction.tile is not None}

    def statistics(self) -> dict[str, typing.Any]:
        """Segmentation statistics over all predictions: pixels per class, active / inactive pixels per
        class (active: input has events), target pixel probabilities (mean, median, min, max)."""
        segmented = [p for p in self.predictions if p.class_counts is not None]
        if not segmented:
            return {"inputs": len(self.predictions), "selected": len(self.selected)}
        classes = max(len(p.class_counts) for p in segmented)
        class_counts = np.zeros(classes, np.int64)
        activity = np.zeros((2, classes), np.int64)
        for prediction in segmented:
            class_counts[: len(prediction.class_counts)] += prediction.class_counts
            activity[:, : prediction.activity_counts.shape[1]] += prediction.activity_counts
        scores = [p.pixels["score"] for p in segmented if p.pixels is not None and len(p.pixels)]
        scores = np.concatenate(scores) if scores else np.zeros(0, np.float32)
        names = [segmented[0].class_name(index) for index in range(classes)]
        return {
            "inputs": len(self.predictions),
            "selected": len(self.selected),
            "pixels": int(class_counts.sum()),
            "active_pixels": int(activity[1].sum()),
            "inactive_pixels": int(activity[0].sum()),
            "class_pixels": dict(zip(names, class_counts.tolist())),
            "active_class_pixels": dict(zip(names, activity[1].tolist())),
            "inactive_class_pixels": dict(zip(names, activity[0].tolist())),
            "target_pixels": int(len(scores)),
            "target_score": None
            if not len(scores)
            else {
                "mean": float(scores.mean()),
                "median": float(np.median(scores)),
                "min": float(scores.min()),
                "max": float(scores.max()),
            },
        }

    def statistics_report(self) -> str:
        """The statistics as text (for debugging a segmentation model)."""
        stats = self.statistics()
        lines = [f"agent {self.agent}: {stats['inputs']} inputs, {stats['selected']} selected"]
        if "pixels" not in stats:
            return lines[0]
        total, active, inactive = stats["pixels"], stats["active_pixels"], stats["inactive_pixels"]
        lines.append(f"pixels: {total}, active: {active}, inactive: {inactive}")

        def share(count, whole):
            return f"{count} ({count / whole:.2%})" if whole else f"{count}"

        for title, key, whole in (
            ("active pixels", "active_class_pixels", active),
            ("inactive pixels", "inactive_class_pixels", inactive),
            ("all pixels", "class_pixels", total),
        ):
            lines.append(f"{title}: " + ", ".join(f"{name} {share(count, whole)}" for name, count in stats[key].items()))
        score = stats["target_score"]
        if score is None:
            lines.append("target pixels: none")
        else:
            lines.append(
                f"target pixels: {stats['target_pixels']}, probability mean {score['mean']:.4f}, "
                f"median {score['median']:.4f}, min {score['min']:.4f}, max {score['max']:.4f}"
            )
        return "\n".join(lines)

    def print_statistics(self):
        print(self.statistics_report(), flush=True)

    def to_dict(self, selected_only: bool = False) -> dict[str, typing.Any]:
        predictions = self.selected if selected_only else self.predictions
        return {
            "agent": self.agent,
            "camera": None if self.frame is None else self.frame.camera,
            "frame": None if self.frame is None else self.frame.name,
            "start_t": None if self.frame is None else self.frame.start_t,
            "end_t": None if self.frame is None else self.frame.end_t,
            "inference_ms": self.inference_ms,
            "predictions": [prediction.to_dict() for prediction in predictions],
        }
