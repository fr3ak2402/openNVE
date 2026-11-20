"""Recordings: reading .raw files (Prophesee / Metavision RAW, EVT 2.0, 2.1, 3.0) like a live camera.

Modules: raw (RawReader, open_raw, read_raw, read_header).
"""

from .raw import RawHeader, RawReader, open_raw, read_header, read_raw

__all__ = ["RawHeader", "RawReader", "open_raw", "read_header", "read_raw"]
