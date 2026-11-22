"""Agent: a model with its configuration, runs on frames and tiles."""

from __future__ import annotations

import time
import typing

import numpy as np

from ..frames import Frame, Tile
from ._util import _resize, _softmax
from .backends import Backend, ModelBackend, ModelInput, create_backend
from .config import AgentConfig, PathLike, load_agent_config, load_agent_configs
from .registry import _TASKS
from .results import AgentResult, lazy_outputs_result
from .tasks import TaskInput



class Agent:
    """A model with its configuration: agent.predict(frame) runs it on the tiles (or the frame).

    The model is loaded on first use, or with agent.load(). An already loaded model (for
    instance a Keras model or a torch.nn.Module created in code) can be passed with model=...
    """

    def __init__(self, config: typing.Union[AgentConfig, dict, str], model: typing.Any = None):
        if isinstance(config, str):
            config = AgentConfig.from_json(config)
        elif isinstance(config, dict):
            config = AgentConfig.from_dict(config)
        config.validate()
        self.config = config
        self._backend: typing.Optional[Backend] = None
        if model is not None:
            self._backend = model if isinstance(model, Backend) else _model_backend(model, config)

    @classmethod
    def from_model(cls, model: typing.Any, name: str = "model", **settings) -> "Agent":
        """Agent for a model object: a torch.nn.Module (PyTorch), or any object with predict(batch),
        for instance a Keras model.
        settings: further agent settings, for instance input="frame" or prediction={...}."""
        return cls(AgentConfig(model=f"<{name}>", name=name, **settings), model=model)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> "Agent":
        """Loads the model now (it is otherwise loaded on first use)."""
        if self._backend is None:
            self._backend = create_backend(self.config)
        return self

    @property
    def backend(self) -> Backend:
        return self.load()._backend

    @property
    def device(self) -> str:
        """Where the model runs: "cpu", "gpu", "npu", "gpu+cpu", "delegate" (loads the model)."""
        return self.backend.device

    def device_description(self) -> str:
        """For example "GPU (batch 8)" or "CPU (GPU: not available or cannot run model.tflite)"."""
        backend = self.backend
        text = backend.device.upper().replace("+CPU", " + CPU")
        if getattr(backend, "device_label", None):
            text += f" {backend.device_label}"
        if backend.device == "delegate":
            library = self.config.delegate["library"] if isinstance(self.config.delegate, dict) else self.config.delegate
            text = f"delegate {library}"
        batch = getattr(backend, "batch_size", None)
        if backend.device != "cpu" and batch:
            text += f" (batch {batch})"
        if backend.device_note:
            text += f" ({backend.device_note})"
        return text

    @property
    def layout(self) -> str:
        """Input layout of the model: "nhwc" or "nchw" (preprocess.layout, "auto": from the backend)."""
        layout = self.config.preprocess.layout
        return self.backend.layout if layout == "auto" else layout

    def _class_softmax(self, values: np.ndarray) -> np.ndarray:
        """Softmax along the class axis: channels of maps (axis 1 for nchw), else the last axis."""
        return _softmax(values, axis=1 if self.layout == "nchw" and values.ndim == 4 else -1)

    @property
    def input(self) -> ModelInput:
        """Model input (shape and dtype)."""
        return self.backend.input

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def __enter__(self) -> "Agent":
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, model={self.config.model!r}, input={self.config.input!r})"

    # preprocessing

    def _model_size(self) -> tuple[typing.Optional[int], typing.Optional[int], typing.Optional[int]]:
        """(height, width, channels) of the model input, None where not fixed."""
        shape = self.input.shape
        if len(shape) != 4:
            return None, None, None
        if self.layout == "nchw":
            return shape[2], shape[3], shape[1]
        return shape[1], shape[2], shape[3]

    def _select_channels(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            data = data[:, :, None]
        channels = self.config.preprocess.channels
        if data.shape[2] != 2:
            return data if channels == "both" else data[:, :, :1]
        if channels == "both":
            return data
        if channels == "on":
            return data[:, :, 1:2]
        if channels == "off":
            return data[:, :, 0:1]
        if channels == "sum":
            return data.sum(axis=2, keepdims=True)
        return data.max(axis=2, keepdims=True)

    def prepare(self, images: typing.Sequence[np.ndarray], pad_to: typing.Optional[tuple[int, int]] = None) -> np.ndarray:
        """Converts frames or tiles (height x width x channels) to a model input batch."""
        preprocess = self.config.preprocess
        model_height, model_width, model_channels = self._model_size()
        if preprocess.resize is not None:
            target = (preprocess.resize[1], preprocess.resize[0])
        elif model_height is not None and model_width is not None:
            target = (model_height, model_width)
        else:
            target = None
        prepared = []
        for image in images:
            image = np.asarray(image, dtype=np.float32)
            image = self._select_channels(image)
            if pad_to is not None and preprocess.pad and image.shape[:2] != pad_to:
                padded = np.zeros(pad_to + image.shape[2:], dtype=np.float32)
                padded[: image.shape[0], : image.shape[1]] = image
                image = padded
            if target is not None:
                image = _resize(image, target[1], target[0])
            prepared.append(image)
        if not prepared:
            height, width = target if target is not None else (pad_to or (0, 0))
            channels = model_channels or (2 if preprocess.channels == "both" else 1)
            batch = np.zeros((0, height or 0, width or 0, channels), dtype=np.float32)
        else:
            shapes = {image.shape for image in prepared}
            if len(shapes) != 1:
                raise ValueError(
                    f"agent {self.name}: inputs of different sizes {sorted(shapes)}; set "
                    "\"preprocess\": {\"pad\": true} or \"resize\": [width, height]"
                )
            batch = np.stack(prepared)
        if model_channels is not None and len(batch) and batch.shape[3] != model_channels:
            raise ValueError(
                f"agent {self.name}: the model expects {model_channels} channel(s), the input has {batch.shape[3]} "
                f"(\"preprocess\": {{\"channels\": ...}}: both = 2 channels, on / off / sum / max = 1 channel)"
            )
        if preprocess.scale != 1.0:
            batch = batch * np.float32(preprocess.scale)
        if self.layout == "nchw":
            batch = batch.transpose(0, 3, 1, 2)
        return np.ascontiguousarray(batch)

    # prediction

    def predict_batch(self, batch: np.ndarray) -> list[np.ndarray]:
        """Runs the model on a prepared batch (in chunks of batch_size), returns the raw outputs."""
        backend = self.backend
        size = self.config.batch_size or len(batch)
        if len(batch) == 0:
            return []
        chunks = [backend.predict(batch[start : start + size]) for start in range(0, len(batch), size)]
        return [np.concatenate([chunk[index] for chunk in chunks]) for index in range(len(chunks[0]))]

    def predict(
        self, data: typing.Union[Frame, typing.Sequence[Tile], np.ndarray]
    ) -> AgentResult:
        """Runs the model on a frame (its tiles or the whole frame, see "input"), a list of
        tiles, or an array (one input of height x width x channels, or a batch)."""
        frame = data if isinstance(data, Frame) else None
        tiles: typing.Optional[list[Tile]] = None
        pad_to = None
        if frame is not None:
            if self.config.input == "tiles":
                if not frame.tiles and frame.tile_rows == 0:
                    raise ValueError(
                        f"agent {self.name} needs tiles: use a frame configuration with \"tiles\" "
                        "(or frame.split(size) first), or set \"input\": \"frame\""
                    )
                tiles = list(frame.tiles)
                size = frame._tile_size or max([tile.width for tile in tiles] + [tile.height for tile in tiles] + [1])
                pad_to = (size, size)
                images = [tile.data for tile in tiles]
            else:
                images = [frame.data]
        elif isinstance(data, np.ndarray):
            images = [data] if data.ndim <= 3 else list(data)
        else:
            tiles = list(data)
            if tiles and not all(isinstance(tile, Tile) for tile in tiles):
                raise TypeError("predict expects a Frame, a list of Tile objects or a numpy array")
            size = max([tile.width for tile in tiles] + [tile.height for tile in tiles] + [1])
            pad_to = (size, size)
            images = [tile.data for tile in tiles]
        batch = self.prepare(images, pad_to)
        start = time.perf_counter()
        outputs = self.predict_batch(batch)
        inference_ms = (time.perf_counter() - start) * 1000
        prediction_config = self.config.prediction
        decoder = _TASKS[prediction_config.task]
        # tasks that read logits (segmentation) get the softmax lazily: only what they use is computed
        logits = bool(outputs) and prediction_config.activation == "softmax" and getattr(decoder, "accepts_logits", False)
        raw_outputs = outputs
        if outputs and prediction_config.activation != "none" and not logits:
            first = outputs[0].astype(np.float32)
            first = self._class_softmax(first) if prediction_config.activation == "softmax" else 1 / (1 + np.exp(-first))
            outputs = [first] + outputs[1:]
        context = TaskInput(
            agent=self,
            frame=frame,
            tiles=tiles,
            images=[np.asarray(image) for image in images],
            pad_to=pad_to,
            outputs=outputs,
            count=len(batch),
            logits=logits,
        )
        predictions = decoder(context)
        if logits:
            return lazy_outputs_result(
                raw_outputs, self._class_softmax, agent=self.name, frame=frame, predictions=predictions, inference_ms=inference_ms
            )
        return AgentResult(
            agent=self.name, frame=frame, predictions=predictions, outputs=outputs, inference_ms=inference_ms
        )


def load_agents(path: PathLike) -> list["Agent"]:
    """Agents of a configuration file. The models are loaded on first use (or with agent.load())."""
    return [Agent(config) for config in load_agent_configs(path)]


def load_agent(path: PathLike, name: typing.Optional[str] = None) -> "Agent":
    """One agent of a configuration file (by name if the file has several)."""
    return Agent(load_agent_config(path, name))


def _model_backend(model, config: AgentConfig) -> Backend:
    """Backend for a model object: torch.nn.Module -> TorchBackend, otherwise predict(batch) / call."""
    from ._torch import TorchBackend, is_torch_module

    if is_torch_module(model):
        return TorchBackend(model, config.threads, config.device, config.fallback, config.gpu_precision,
                            batch_size=config.batch_size)
    return ModelBackend(model)
