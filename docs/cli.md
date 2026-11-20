# Command line

After installation, the `opennve` command is available (or `python -m opennve`).

## `opennve list`

Shows the connected cameras.

```sh
opennve list
opennve list --config cameras.json   # with the names from the configuration
opennve list --json                  # as JSON
```

```text
SE1-S4     05110634   USB 3.0 SuperSpeed (5.0 Gb/s)    ok (left)
SE1-S4     05110636   USB 3.0 SuperSpeed (5.0 Gb/s)    ok (right)
```

If an error is shown instead of `ok`, the camera cannot be used, see [Troubleshooting](troubleshooting.md).

## `opennve template <model>`

Prints a complete configuration with all default values.

```sh
opennve template SE1-S4
opennve template SE1-S4 --serial 05110634 --name left > left.json
```

## `opennve frame-template`

Prints a frame configuration file (`{"frames": [...]}`) with one configuration, see [Frames and tiles](frames.md).

```sh
opennve frame-template --name tiles80 --width 1280 --height 720 --tiles 80 > frames.json
opennve frame-template --name recent --decay exponential
opennve frame-template --name binary --mode binary
```

Without `--width` / `--height` the sensor size is used, without `--tiles` the frame is not split.

## `opennve check <file> [<file> ...]`

Validates camera, frame and agent configuration files without opening a camera. `--load` also loads the agent models and prints their input shape (needs LiteRT or TensorFlow, see [Agents](agents.md)).

```sh
opennve check cameras.json frames.json
opennve check agents.json --load
```

```text
cameras.json
  camera left: SE1-S4 05110634
  camera right: SE1-S4 05110636
  configuration is valid
frames.json
  frame full: 1280 x 720, time_surface (linear), no tiles
  frame tiles80: 1280 x 720, time_surface (linear), tiles 80 px
  configuration is valid
```

Errors name the camera and the field, for instance `cameras.json: camera 1: settings.clock must be one of internal, internal_with_output_enabled, external`. The exit code is then `1`.

## `opennve driver`

Shows or installs the USB driver of the connected cameras (WinUSB on Windows).

```sh
opennve driver                              # state of all cameras
opennve driver --install                    # install WinUSB where no driver is installed
opennve driver --install --model SE1-S4     # only SE1-S4 cameras
opennve driver --install --force            # also replace another (vendor) driver
```

```text
SENSING SE1-S4-USB (04B4:00C4) driver: WinUSB | ready
```

The exit code is `1` if a camera is not ready.

## `opennve raw <file>`

Shows the header and statistics of a RAW file: format, sensor size, serial number, number of events, time span and event rate.

```sh
opennve raw recording.raw
opennve raw recording.raw --json
opennve raw camera_data.bin --format EVT3 --width 1280 --height 720   # file without header
```

## `opennve models`

Lists the supported models.

## Commands of the embedded driver

| Command | Description |
|---|---|
| `ndls` | Connected devices (table from `neuromorphic_drivers`) |
| `neuromorphic-drivers-install-windows-driver` | Installs the WinUSB driver on Windows (`--list`, `--device VID:PID`, `--force`) |
| `neuromorphic-drivers-install-udev-rules` | Installs udev rules on Linux |
