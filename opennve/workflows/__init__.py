"""Workflows: freely configurable chains of steps (source -> processing steps -> sink), run in threads.

See engine.py for the configuration format and steps.py for the built-in steps.
"""

from .engine import (
    StepStatistics,
    Workflow,
    WorkflowConfig,
    load_workflow,
    load_workflow_config,
    load_workflow_configs,
    save_workflow_configs,
    workflow_configs_from_data,
)
from .item import WorkItem
from .steps import Step, StepContext, list_steps, register_step, sub_window, window_bounds

__all__ = [
    "Step",
    "StepContext",
    "StepStatistics",
    "WorkItem",
    "Workflow",
    "WorkflowConfig",
    "list_steps",
    "load_workflow",
    "load_workflow_config",
    "load_workflow_configs",
    "register_step",
    "save_workflow_configs",
    "sub_window",
    "window_bounds",
    "workflow_configs_from_data",
]
