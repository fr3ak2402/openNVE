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

## `opennve check <file>`

Validates a configuration file without opening a camera.

```sh
opennve check cameras.json
```

```text
left: SE1-S4 05110634
right: SE1-S4 05110636
2 camera(s), configuration is valid
```

Errors name the camera and the field, for instance `cameras.json: camera 1: settings.clock must be one of internal, internal_with_output_enabled, external`. The exit code is then `1`.

## `opennve models`

Lists the supported models.

## Commands of the embedded driver

| Command | Description |
|---|---|
| `ndls` | Connected devices (table from `neuromorphic_drivers`) |
| `neuromorphic-drivers-install-windows-driver` | Installs the WinUSB driver on Windows (`--list`, `--device VID:PID`, `--force`) |
| `neuromorphic-drivers-install-udev-rules` | Installs udev rules on Linux |
