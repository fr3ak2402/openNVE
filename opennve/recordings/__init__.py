"""Recordings: reading .raw files (Prophesee / Metavision RAW, EVT 2.0, 2.1, 3.0) and HDF5 event
files (.h5 / .hdf5 with the columns x, y, p, t and optionally l) like a live camera.

Modules: reader (RecordingReader, the common base), raw (RawReader, open_raw), hdf5 (Hdf5Reader,
open_hdf5, Hdf5Writer, write_hdf5), plus open_recording (chooses the reader by the file type).
"""

from .hdf5 import HDF5_SUFFIXES, Hdf5Reader, Hdf5Writer, is_hdf5, open_hdf5, open_recording, read_hdf5, write_hdf5
from .raw import RawHeader, RawReader, open_raw, read_header, read_raw
from .reader import RecordingReader

__all__ = [
    "HDF5_SUFFIXES",
    "Hdf5Reader",
    "Hdf5Writer",
    "RawHeader",
    "RawReader",
    "RecordingReader",
    "is_hdf5",
    "open_hdf5",
    "open_raw",
    "open_recording",
    "read_hdf5",
    "read_header",
    "read_raw",
    "write_hdf5",
]
