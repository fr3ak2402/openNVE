"""Running agents on every time window of a camera or RAW file."""

from __future__ import annotations

import dataclasses
import pathlib
import typing

from ..frames import Frame, FrameConfig, as_frame_configs
from .agent import Agent, load_agents
from .config import AgentConfig
from .results import AgentResult


def as_agents(agents) -> list[Agent]:
    """List of Agent from a path, an Agent, an AgentConfig, a dict or a list of these."""
    if isinstance(agents, (str, pathlib.Path)) and not str(agents).lstrip().startswith("{"):
        return load_agents(agents)
    if isinstance(agents, (Agent, AgentConfig, dict, str)):
        agents = [agents]
    result = [agent if isinstance(agent, Agent) else Agent(agent) for agent in agents]
    names = [agent.name for agent in result]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate agent name(s): {', '.join(duplicates)}")
    return result



@dataclasses.dataclass
class AgentStep:
    """Results of all agents for one time window.

    Attributes:
        window: The time window (window.events: its events).
        frames: {frame configuration name: Frame} built for the agents.
        results: {agent name: AgentResult}.
    """

    window: typing.Any
    frames: dict[str, Frame]
    results: dict[str, AgentResult]

    def __getitem__(self, agent: str) -> AgentResult:
        return self.results[agent]


def _frame_assignments(agents: list[Agent], frame_configs: list[FrameConfig]) -> dict[str, FrameConfig]:
    by_name = {config.name: config for config in frame_configs}
    assignments = {}
    for agent in agents:
        name = agent.config.frame
        if name is None:
            if len(frame_configs) != 1:
                raise ValueError(
                    f"agent {agent.name}: set \"frame\" to one of {', '.join(by_name)} "
                    "(several frame configurations are given)"
                )
            assignments[agent.name] = frame_configs[0]
        elif name not in by_name:
            raise ValueError(
                f"agent {agent.name}: no frame configuration named {name!r} (available: {', '.join(by_name)})"
            )
        else:
            assignments[agent.name] = by_name[name]
    return assignments


def process_window(window, agents, frame_configs=None) -> AgentStep:
    """Builds the frames needed by the agents for one window and runs the agents."""
    agents = as_agents(agents)
    frame_configs = _as_frame_config_list(frame_configs)
    assignments = _frame_assignments(agents, frame_configs)
    return _process(window, agents, assignments)


def _process(window, agents: list[Agent], assignments: dict[str, FrameConfig]) -> AgentStep:
    frames: dict[str, Frame] = {}
    results: dict[str, AgentResult] = {}
    for agent in agents:
        config = assignments[agent.name]
        if config.name not in frames:
            frames[config.name] = window.frame(config)
        results[agent.name] = agent.predict(frames[config.name])
    return AgentStep(window=window, frames=frames, results=results)


def _as_frame_config_list(frame_configs) -> list[FrameConfig]:
    if frame_configs is None:
        return [FrameConfig(name="frame1")]
    if isinstance(frame_configs, (str, pathlib.Path)) and not str(frame_configs).lstrip().startswith("{"):
        from ..frames import load_frame_configs

        return load_frame_configs(frame_configs)
    return as_frame_configs(frame_configs)


def run_agents(
    source, agents, frame_configs=None, duration_ms: typing.Optional[float] = None
) -> typing.Iterator[AgentStep]:
    """Runs agents on every time window of a camera or a RAW file.

    source: an open EventCamera or RawReader (anything with windows(duration_ms)).
    agents: list of Agent (load_agents("agents.json")), agent configurations, or a file path.
    frame_configs: frame configurations (load_frame_configs("frames.json")) or a file path.
        Each agent uses the configuration named in its "frame" setting; each frame is built
        once per window, also if several agents use it.
    Yields one AgentStep per window: step.results[agent name], step.frames, step.window.
    """
    agents = as_agents(agents)
    frame_configs = _as_frame_config_list(frame_configs)
    assignments = _frame_assignments(agents, frame_configs)
    for agent in agents:
        agent.load()
    for window in source.windows(duration_ms):
        yield _process(window, agents, assignments)


