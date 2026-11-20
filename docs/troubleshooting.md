# Troubleshooting

## Camera not found

`opennve list` shows `No camera found`, or `ndls` shows an empty table.

1. **Does the operating system detect the camera?**
   - Windows: Device Manager → camera → *Properties* → *Details* → *Hardware Ids*. For the SE1-S4, it must show `USB\VID_04B4&PID_00C4`.
   - Linux: `lsusb | grep 04b4`
   - If the camera does not appear, check the cable, port and power supply.
2. **Windows: is a driver installed?**
   ```powershell
   neuromorphic-drivers-install-windows-driver --list
   ```
   If it shows `driver: (none)` or a driver other than `WinUSB`, install WinUSB with `opennve driver --install` (or `opennve.ensure_driver()` in Python). Without a driver, the camera is invisible to programs.
3. **Is Python running in WSL, Docker or a virtual machine?** USB is not passed through by default there (on WSL, for instance, use `usbipd attach`).

## Camera found, but with an error

- **Linux, permission denied:** run `neuromorphic-drivers-install-udev-rules`, run the `sudo` command it prints, then reconnect the camera.
- **`unsupported sensor`:** the sensor reports an unexpected ID. Please open an issue with the output.

## `pip install` fails with `UnicodeDecodeError`

The wheel was installed with `-r`. Install it without `-r`:

```sh
pip install opennve-1.0.0b0-cp314-cp314-win_amd64.whl
```

## `No module named neuromorphic_drivers.sensing_se1_s4` or wrong driver version

A separately installed `neuromorphic_drivers` (for instance from PyPI) overwrites the embedded version:

```sh
pip uninstall -y neuromorphic_drivers opennve
pip install opennve-...whl
python -c "import opennve; print(opennve.NEUROMORPHIC_DRIVERS_VERSION)"
```

## Strong noise

1. **Check for flickering light:** LED and fluorescent lamps flicker at 100 Hz (50 Hz mains) and produce events over large areas. Test with daylight or an incandescent lamp.
2. **More light:** event cameras are noisy in dark scenes.
3. **Increase the biases:** `"biases": {"diff_on": 30, "diff_off": 30}`, adjust step by step (see [Configuration](configuration.md#biases-se1-s4)).
4. **Enable the noise filter:** `"filter": {"background_activity": {"enabled": true}}`. Filter more with a smaller `time_us` or `min_neighbors: 2`.

## High event rates, latency, lost data

If `packet.status.ring.backlog` keeps growing, processing is slower than the camera.

- Produce fewer events: increase the biases `diff_on`/`diff_off`, restrict the image area with masks, avoid flickering light.
- Rate limiter in the sensor: `"rate_limiter": {"reference_period_us": 200, "maximum_events_per_period": 4000}`
- Faster processing: work with NumPy arrays (`packet.events`) instead of `rows()` or `to_csv()` per event.
- Display less: show frames less often. Displaying with OpenCV often takes more time than reading.

## Multiple cameras: timestamps drift apart

Without synchronization, each camera has its own clock. Solution: a sync cable and the `clock` setting, see [Multiple cameras](multiple-cameras.md#synchronizing-clocks).

## Configuration rejected

`opennve check file.json` shows the cause with the affected field. Common cases:

| Message | Cause |
|---|---|
| `serial must be a string` | Write the serial number in quotes: `"05110634"` |
| `unknown setting(s) in settings` | Typo in a field name. `opennve template <model>` shows the valid names |
| `output field c is not supported` | The camera name is stored once per packet in `packet.camera`, not in the events |
| `duplicate camera name` | Each `name` must be unique within a file |
