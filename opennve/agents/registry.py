"""Registries of model runtimes (backends) and output decoders (tasks).

New runtimes: register_backend("onnx", factory, suffixes=[".onnx"]) with factory(config) -> Backend
(plugins do this in their register(), see opennve.plugins).
New tasks: register_task("detection", decoder) with decoder(TaskInput) -> list[Prediction].
"""

from __future__ import annotations

import typing

_BACKENDS: dict[str, typing.Callable[..., typing.Any]] = {}  # name -> factory(config)
_BACKEND_SUFFIXES: dict[str, str] = {}  # model file suffix -> backend name
_BACKEND_OPTIONS: dict[str, dict[str, typing.Any]] = {}  # name -> {"threads", "accelerator"}
_TASKS: dict[str, typing.Callable[..., list]] = {}  # name -> decoder


def register_backend(name: str, factory=None, suffixes: typing.Sequence[str] = (), *, threads: str = "self",
                     accelerator: bool = False):
    """Registers a model runtime: factory(config: AgentConfig) -> Backend.
    suffixes: model file suffixes chosen by backend "auto", for instance [".onnx"].
    threads: "planned": workflows run one model per thread and split the cores between them
    (like TensorFlow Lite); "self": the runtime uses the cores itself (one workflow thread).
    accelerator: the runtime always runs on an accelerator (workflows give it 2 threads).
    Usable as a decorator: @register_backend("onnx", suffixes=[".onnx"])."""
    if threads not in ("planned", "self"):
        raise ValueError("threads must be planned or self")

    def decorator(function):
        _BACKENDS[name] = function
        _BACKEND_OPTIONS[name] = {"threads": threads, "accelerator": accelerator}
        for suffix in suffixes:
            _BACKEND_SUFFIXES[suffix.lower()] = name
        return function

    return decorator if factory is None else decorator(factory)


def list_backends() -> list[str]:
    """Names of the registered backends (with the backends of the installed plugins)."""
    from ..plugins import load_plugins

    load_plugins()
    return sorted(_BACKENDS)


def backend_options(name: str) -> dict[str, typing.Any]:
    """{"threads": "planned" | "self", "accelerator": bool} of a registered backend."""
    return dict(_BACKEND_OPTIONS.get(name, {"threads": "self", "accelerator": False}))


def register_task(name: str, decoder=None):
    """Registers a task: decoder(TaskInput) -> list[Prediction] (one per input).
    Usable as a decorator: @register_task("detection")."""

    def wrapper(function):
        _TASKS[name] = function
        return function

    return wrapper if decoder is None else wrapper(decoder)


def list_tasks() -> list[str]:
    return sorted(_TASKS)
