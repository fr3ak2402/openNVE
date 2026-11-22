"""Compares the devices (CPU, GPU, NPU) for an agent: speed and agreement with the CPU.

    results = opennve.benchmark_agent(config, devices=("cpu", "gpu", "npu"), frames=frames)
    print(opennve.benchmark_report(results))

On the command line: opennve benchmark agents.json --agent prototyp --recording data/rec.h5
--frames frames.json.
"""

from __future__ import annotations

import time
import typing

import numpy as np

from .agent import Agent
from .config import AgentConfig


def _inputs(config: AgentConfig, frames, count: int, seed: int) -> np.ndarray:
    agent = Agent(config.copy(device="cpu", delegate=None, fallback=True))
    if frames:
        images, pad_to = [], None
        for frame in frames:
            if config.input == "tiles":
                size = frame._tile_size or max([tile.width for tile in frame.tiles] + [tile.height for tile in frame.tiles] + [1])
                pad_to = (size, size)
                images.extend(tile.data for tile in frame.tiles)
            else:
                images.append(frame.data)
        if images:
            batch = agent.prepare(images[:count] if count else images, pad_to)
            agent.close()
            return batch
    shape = tuple(size or 1 for size in agent.input.shape[1:])
    agent.close()
    rng = np.random.default_rng(seed)
    data = rng.random((count or 64,) + shape)
    return data.astype(np.float32)


def benchmark_agent(
    config: AgentConfig,
    devices: typing.Sequence[str] = ("cpu", "gpu", "npu"),
    frames: typing.Optional[typing.Sequence] = None,
    count: int = 0,
    repeat: int = 3,
    seed: int = 0,
) -> list[dict[str, typing.Any]]:
    """Runs the agent on the same inputs on every device.

    frames: frames to take the inputs from (their tiles, or the whole frames for "input":
    "frame"); without frames random inputs of the model's shape are used (count, default 64).
    Returns one dict per device: requested, device (where it really ran), description, note (why
    another device was used), batch_size, load_ms,
    ms_per_input, inputs_per_second, agreement (share of equal arg-max classes compared with the
    first device, 0..1), error.
    """
    batch = _inputs(config, frames, count, seed)
    results = []
    reference = None
    for requested in devices:
        result: dict[str, typing.Any] = {"requested": requested, "inputs": len(batch)}
        agent = Agent(config.copy(device=requested, fallback=True))
        try:
            start = time.perf_counter()
            agent.load()
            result["load_ms"] = (time.perf_counter() - start) * 1000
            result["device"] = agent.device
            result["description"] = agent.device_description()
            result["note"] = agent.backend.device_note
            result["batch_size"] = getattr(agent.backend, "batch_size", None) or config.batch_size
            agent.predict_batch(batch[: max(1, getattr(agent.backend, "batch_size", 1))])  # warm up
            best = None
            outputs = None
            # in chunks like the frames of a workflow: one call with thousands of inputs would
            # need gigabytes for the intermediate tensors
            chunk = config.batch_size or 128
            for _ in range(max(1, repeat)):
                start = time.perf_counter()
                parts = [agent.predict_batch(batch[index : index + chunk])[0] for index in range(0, len(batch), chunk)]
                elapsed = time.perf_counter() - start
                outputs = [np.concatenate(parts)]
                best = elapsed if best is None else min(best, elapsed)
            result["ms_per_input"] = best * 1000 / len(batch)
            result["inputs_per_second"] = len(batch) / best if best else None
            axis = 1 if agent.layout == "nchw" and outputs and outputs[0].ndim == 4 else -1  # class axis
            classes = np.argmax(outputs[0], axis=axis) if outputs and outputs[0].ndim > 1 else None
            if reference is None:
                reference = classes
                result["agreement"] = 1.0 if classes is not None else None
            elif classes is not None and reference is not None and classes.shape == reference.shape:
                result["agreement"] = float(np.mean(classes == reference))
            else:
                result["agreement"] = None
            result["error"] = None
        except Exception as error:  # report and go on with the next device
            result.update(device=None, description=None, ms_per_input=None, inputs_per_second=None, agreement=None,
                          error=str(error))
        finally:
            agent.close()
        results.append(result)
    return results


def benchmark_report(results: list[dict[str, typing.Any]]) -> str:
    lines = [f"{'device':<8} {'runs on':<10} {'batch':>5} {'ms/input':>9} {'inputs/s':>9} {'agreement':>10} {'load':>8}"]
    notes = []
    for result in results:
        if result.get("error"):
            lines.append(f"{result['requested']:<8} error")
            notes.append(f"{result['requested']}: {result['error']}")
            continue
        agreement = "-" if result["agreement"] is None else f"{result['agreement'] * 100:.2f} %"
        lines.append(
            f"{result['requested']:<8} {result['device']:<10} {result.get('batch_size') or '-':>5} "
            f"{result['ms_per_input']:>9.2f} {result['inputs_per_second']:>9.0f} {agreement:>10} {result['load_ms']:>6.0f}ms"
        )
        if result.get("note"):
            notes.append(f"{result['requested']}: {result['note']}")
    inputs = results[0]["inputs"] if results else 0
    lines.append(f"{inputs} inputs; agreement: equal classes compared with {results[0]['requested'] if results else '-'}")
    lines.extend(notes)
    return "\n".join(lines)
