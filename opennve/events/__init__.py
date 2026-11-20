"""Event processing: noise filters, output formats (x, y, p, t) and time windows.

Modules: filters (FilterConfig, background activity filter), output (OutputConfig, CSV),
window (WindowConfig, EventWindow, WindowAccumulator).
"""

from .filters import BackgroundActivityConfig, BackgroundActivityFilter, EventFilter, FilterConfig
from .output import OutputConfig, to_csv, to_rows
from .window import EventWindow, WindowAccumulator, WindowConfig

__all__ = [
    "BackgroundActivityConfig",
    "BackgroundActivityFilter",
    "EventFilter",
    "EventWindow",
    "FilterConfig",
    "OutputConfig",
    "WindowAccumulator",
    "WindowConfig",
    "to_csv",
    "to_rows",
]
