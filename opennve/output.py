"""Output format of the events, configured with the "output" section of a camera configuration.

JSON example:

    "output": {"fields": ["c", "x", "y", "p", "t"]}

Fields (any order, each at most once):

    c: camera name (from the configuration)
    x: pixel column
    y: pixel row
    p: polarity (1 = ON, brighter / 0 = OFF, darker)
    t: timestamp in microseconds
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

FIELDS = {
    "c": "camera name",
    "x": "pixel column",
    "y": "pixel row",
    "p": "polarity (1 = ON, 0 = OFF)",
    "t": "timestamp in microseconds",
}
DEFAULT_FIELDS = ("c", "x", "y", "p", "t")

_NUMERIC_DTYPES = {"x": np.uint16, "y": np.uint16, "p": np.uint8, "t": np.uint64}


@dataclasses.dataclass
class OutputConfig:
    """Fields of the output events, in order."""

    fields: list[str] = dataclasses.field(default_factory=lambda: list(DEFAULT_FIELDS))

    def validate(self):
        if not isinstance(self.fields, list) or len(self.fields) == 0:
            raise ValueError("output.fields must be a non-empty list, for instance [\"c\", \"x\", \"y\", \"p\", \"t\"]")
        unknown = [field for field in self.fields if field not in FIELDS]
        if unknown:
            raise ValueError(
                f"unknown output field(s): {', '.join(map(str, unknown))} "
                f"(supported: {', '.join(f'{key} = {value}' for key, value in FIELDS.items())})"
            )
        duplicates = sorted({field for field in self.fields if self.fields.count(field) > 1})
        if duplicates:
            raise ValueError(f"duplicate output field(s): {', '.join(duplicates)}")

    def to_dict(self) -> dict[str, typing.Any]:
        return {"fields": list(self.fields)}

    @classmethod
    def from_dict(cls, data: typing.Optional[dict[str, typing.Any]]) -> "OutputConfig":
        data = dict(data or {})
        unknown = set(data) - {"fields"}
        if unknown:
            raise ValueError(f"unknown output setting(s): {', '.join(sorted(unknown))}")
        fields = data.get("fields", list(DEFAULT_FIELDS))
        if isinstance(fields, str):
            # "c,x,y,p,t" is accepted as well
            fields = [field.strip() for field in fields.split(",") if field.strip()]
        result = cls(fields=list(fields) if isinstance(fields, (list, tuple)) else fields)
        result.validate()
        return result

    def dtype(self, camera_name: str) -> np.dtype:
        return np.dtype(
            [
                (field, f"<U{max(1, len(camera_name))}" if field == "c" else _NUMERIC_DTYPES[field])
                for field in self.fields
            ]
        )

    def format(self, events: np.ndarray, camera_name: str) -> np.ndarray:
        """Polarity events (t, x, y, on) -> structured array with the configured fields, in order."""
        result = np.empty(len(events), dtype=self.dtype(camera_name))
        for field in self.fields:
            if field == "c":
                result["c"] = camera_name
            elif field == "p":
                result["p"] = events["on"]
            else:
                result[field] = events[field]
        return result


def to_rows(events: np.ndarray) -> list[tuple]:
    """Structured output array -> list of tuples with Python values, for instance [("left", 12, 40, 1, 1003)]."""
    return events.tolist()


def to_csv(events: np.ndarray, header: bool = False, delimiter: str = ",") -> str:
    """Structured output array -> CSV text (one event per line)."""
    lines = []
    if header:
        lines.append(delimiter.join(events.dtype.names))
    lines.extend(delimiter.join(str(value) for value in row) for row in events.tolist())
    return "\n".join(lines) + ("\n" if lines else "")
