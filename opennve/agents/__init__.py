"""Agents: run TensorFlow, TensorFlow Lite and PyTorch models on frames and tiles.

An agent combines a model with the settings needed to feed it: which input it gets (the tiles
of a frame, or the whole frame), which frame configuration builds that input, how the data is
prepared and how the outputs are read. Agent configurations are kept in their own file, like
the camera and frame configurations:

    {
        "agents": [
            {
                "name": "insects",
                "model": "models/insects.tflite",
                "backend": "auto",
                "input": "tiles",
                "frame": "tiles80",
                "batch_size": 64,
                "threads": 2,
                "preprocess": {"channels": "both", "pad": true, "resize": null, "layout": "auto", "scale": 1.0},
                "prediction": {"activation": "none", "labels": ["background", "insect"], "threshold": 0.5}
            }
        ]
    }

Backends (backend "auto" chooses by the model path):

    tflite:     .tflite files, run with LiteRT (pip install ai-edge-litert), tflite_runtime
                or tf.lite (TensorFlow)
    tensorflow: Keras models (.keras, .h5) run with model.predict(...), and SavedModel
                directories run with their serving signature (pip install "opennve[tensorflow]")
    pytorch:    PyTorch models: .pt2 (torch.export), .pt / .pth (TorchScript; complete models
                saved with torch.save(model) with "safe_mode": false), or an nn.Module passed
                with Agent.from_model (pip install "opennve[pytorch]"); inputs channels first

Use:

    agents = opennve.load_agents("agents.json")
    frame_configs = opennve.load_frame_configs("frames.json")
    for step in opennve.run_agents(camera, agents, frame_configs):
        for prediction in step.results["insects"].selected:
            print(prediction.tile.id, prediction.label, prediction.score)

or, for one frame: result = agent.predict(frame).

Modules: config (AgentConfig and files), backends (TensorFlow Lite, TensorFlow, model objects), _torch (PyTorch),
tasks (classification, segmentation), results (Prediction, AgentResult), agent (Agent),
runner (run_agents), registry (register_backend, register_task).
"""

from .agent import Agent, load_agent, load_agents
from ._litert import AcceleratorUnavailable, CompiledModelBackend, litert_accelerators, static_model
from ._torch import TorchBackend, torch_info
from .backends import Backend, ModelBackend, ModelInput, TensorFlowBackend, TFLiteBackend, create_backend
from .benchmark import benchmark_agent, benchmark_report
from .config import (
    AgentConfig,
    PredictionConfig,
    PreprocessConfig,
    agent_configs_from_data,
    load_agent_config,
    load_agent_configs,
    save_agent_configs,
)
from .registry import list_backends, list_tasks, register_backend, register_task
from .results import AgentResult, Prediction
from .runner import AgentStep, as_agents, process_window, run_agents
from .tasks import TaskInput

__all__ = [
    "AcceleratorUnavailable",
    "CompiledModelBackend",
    "benchmark_agent",
    "benchmark_report",
    "litert_accelerators",
    "static_model",
    "torch_info",
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentStep",
    "Backend",
    "ModelBackend",
    "ModelInput",
    "Prediction",
    "PredictionConfig",
    "PreprocessConfig",
    "TFLiteBackend",
    "TaskInput",
    "TensorFlowBackend",
    "TorchBackend",
    "agent_configs_from_data",
    "as_agents",
    "create_backend",
    "list_backends",
    "list_tasks",
    "load_agent",
    "load_agent_config",
    "load_agent_configs",
    "load_agents",
    "process_window",
    "register_backend",
    "register_task",
    "run_agents",
    "save_agent_configs",
]
