"""Registries of model runtimes (backends) and output decoders (tasks).

New runtimes: register_backend("onnx", factory, suffixes=[".onnx"]) with factory(config) -> Backend.
New tasks: register_task("detection", decoder) with decoder(TaskInput) -> list[Prediction].
"""

from __future__ import annotations

import typing

_BACKENDS: dict[str, typing.Callable[..., typing.Any]] = {}  # name -> factory(config)
_BACKEND_SUFFIXES: dict[str, str] = {}  # model file suffix -> backend name
_TASKS: dict[str, typing.Callable[..., list]] = {}  # name -> decoder


def register_backend(name: str, factory=None, suffixes: typing.Sequence[str] = ()):
    """Registers a model runtime: factory(config: AgentConfig) -> Backend.
    suffixes: model file suffixes chosen by backend "auto", for instance [".onnx"].
    Usable as a decorator: @register_backend("onnx", suffixes=[".onnx"])."""

    def decorator(function):
        _BACKENDS[name] = function
        for suffix in suffixes:
            _BACKEND_SUFFIXES[suffix.lower()] = name
        return function

    return decorator if factory is None else decorator(factory)


def list_backends() -> list[str]:
    return sorted(_BACKENDS)


def register_task(name: str, decoder=None):
    """Registers a task: decoder(TaskInput) -> list[Prediction] (one per input).
    Usable as a decorator: @register_task("detection")."""

    def wrapper(function):
        _TASKS[name] = function
        return function

    return wrapper if decoder is None else wrapper(decoder)


def list_tasks() -> list[str]:
    return sorted(_TASKS)
