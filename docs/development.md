# Development

## Project structure

```text
opennve/
├── __init__.py      public API
├── __main__.py      command line (opennve ...)
├── camera.py        EventCamera, CameraPacket, open_camera, open_cameras
├── config.py        CameraConfig, loading/saving JSON, validation of the hardware settings
├── devices.py       SE1S4Driver, EVK4Driver, EVK3HDDriver
├── discovery.py     list_cameras, CameraInfo
├── driver.py        driver_status, install_driver, ensure_driver (WinUSB)
├── filters.py       FilterConfig, noise filter, polarity filter
├── models.py        supported models (from neuromorphic_drivers)
├── output.py        OutputConfig (fields x, y, p, t), CSV
├── raw.py           RAW files: header, EVT 2.0 / 2.1 / 3.0 decoders, RawReader
└── window.py        WindowConfig, WindowAccumulator, EventWindow, frames
examples/            cameras.json, viewer.py, frames.py, list_cameras.py, raw_player.py
tests/               pytest, without hardware (the camera is simulated)
tests/data/          RAW reference files written by faery (independent encoder)
tools/bundle_wheel.py  embeds neuromorphic_drivers into the wheels
tools/docs_to_wiki.py  publishes docs/ to the GitHub wiki
docs/                this documentation
```

## Tests

```sh
pip install -e ".[test]"
pytest
```

The tests do not need a camera: `neuromorphic_drivers.open` and `list_devices` are replaced by fakes. They cover configuration and validation, filters, output format, time windows, frames, the camera class and the command line.

## Hardware settings of new models

`settings` is not maintained by hand. `config.py` reads the fields and types directly from the `Configuration` dataclasses of `neuromorphic_drivers` (nested dataclasses, enums, `Optional`, integer types with value ranges). New models or fields in the driver are therefore available automatically.

## Building wheels (GitHub Actions)

`.github/workflows/build.yml` runs on every push to `main`, on pull requests and manually (*Actions → Build → Run workflow*):

1. `neuromorphic_drivers` is built from [fr3ak2402/neuromorphic-drivers](https://github.com/fr3ak2402/neuromorphic-drivers) for Python 3.14 and 3.15 on Windows x64, Linux x64/ARM64 and macOS ARM/Intel.
2. `tools/bundle_wheel.py` merges each driver wheel with the `opennve` package into one wheel: same platform tags, console scripts and licenses of both packages.
3. Each wheel is installed in a fresh environment and tested with all tests (`.github/actions/test-wheel`).
4. The wheels are artifacts of the run. For tags `v*`, they are also attached to a GitHub release.

`NEUROMORPHIC_DRIVERS_REF` in the workflow selects the driver version that is embedded (`main` by default). For reproducible releases, use a tag or commit.

Bundling locally:

```sh
python tools/bundle_wheel.py path/to/neuromorphic_drivers-wheels/ --out dist
```

## Wiki

`.github/workflows/wiki.yml` publishes `docs/` to the [GitHub wiki](https://github.com/fr3ak2402/openNVE/wiki) whenever `docs/` changes on `main` (or manually: *Actions → Wiki → Run workflow*). `tools/docs_to_wiki.py` turns `docs/README.md` into the wiki home page, gives the other pages readable names, rewrites the links and generates the sidebar.

- Edit the documentation in `docs/` only. Generated wiki pages are overwritten on the next run; pages created directly in the wiki are kept.
- Before the first run, the wiki must exist: open the *Wiki* tab of the repository and create the first page (any content).
- New pages in `docs/` appear in the wiki automatically. For a nicer title and position in the sidebar, add them to `PAGES` in `tools/docs_to_wiki.py`.

## Release

1. Increase the version in `pyproject.toml` (`version = "0.12.2"`).
2. Commit and push.
3. Create a tag:
   ```sh
   git tag v0.12.2
   git push --tags
   ```
4. The action builds all wheels and attaches them to the release `v0.12.2`.

## New Python version

The driver is written in Rust and compiled separately for each Python version. For a new version, for instance 3.16:

- add it to `interpreters` (Linux) and `python-version` in `build.yml`,
- add a test step `Test (Python 3.16)`.

## Licenses

`neuromorphic_drivers` is distributed under the MIT License. Its license is included in every wheel at `opennve-*.dist-info/licenses/neuromorphic_drivers/LICENSE`. The WinUSB installer (`winusb_installer.exe`, LGPL-3.0) is not distributed with openNVE, it is downloaded on demand.
