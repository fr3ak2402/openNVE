# Installation

## Requirements

- Python **3.14 or newer**
- Windows x64, Linux x64, Linux ARM64 (for instance a Raspberry Pi 5 with a 64-bit OS) or macOS
- An event camera on a **USB 3 port** (up to 1.6 Gb/s for the SE1-S4)

## Installing the wheel

Download the wheel for your platform from the [releases](https://github.com/fr3ak2402/openNVE/releases) or from the artifacts of the *Build* GitHub Action.

| File name contains | Platform |
|---|---|
| `win_amd64` | Windows x64 |
| `manylinux_2_28_x86_64` | Linux x64 |
| `manylinux_2_28_aarch64` | Linux ARM64 (Raspberry Pi 5) |
| `macosx_..._arm64` | macOS Apple Silicon |
| `macosx_..._x86_64` | macOS Intel |

`cp314` means Python 3.14, `cp315` means Python 3.15. `python --version` shows your version.

```sh
pip uninstall -y neuromorphic_drivers     # only if installed separately (see below)
pip install opennve-0.25.9-cp314-cp314-win_amd64.whl
```

pip installs numpy and OpenCV automatically. Reading `.raw` files and building frames only needs numpy. To pass events as pandas DataFrames (for instance to `FrameEngine.generate_frame`), install the optional extra:

```sh
pip install "opennve-0.25.9-cp314-cp314-win_amd64.whl[pandas]"
```

> **Important:** The wheel contains `neuromorphic_drivers` 0.19.2 with the SE1-S4 driver. A separately installed `neuromorphic_drivers` would overwrite it (and vice versa), so uninstall it first.

> **Note:** Install the wheel **without** `-r`. `pip install -r file.whl` reads the wheel as a text file and fails with `UnicodeDecodeError`.

Check the installation:

```sh
python -c "import opennve; print(opennve.__version__, opennve.NEUROMORPHIC_DRIVERS_VERSION)"
opennve models
```

## Camera drivers

### Windows

Windows only lets programs access a USB device that has a driver. openNVE uses Microsoft's **WinUSB** driver. Run once per computer and camera model, with the camera connected:

```powershell
neuromorphic-drivers-install-windows-driver
```

Confirm the administrator prompt. The installation can take a few minutes. The command does the same as [Zadig](https://zadig.akeo.ie).

```powershell
neuromorphic-drivers-install-windows-driver --list   # show cameras and their current driver
neuromorphic-drivers-install-windows-driver --force  # replace an existing (vendor) driver
```

Cameras with another driver (for instance Metavision's) are not changed without `--force`. With WinUSB, the vendor's software may stop working. To revert, use *Driver → Roll Back Driver* in the Device Manager.

From Python, for instance when your program starts (only installs if needed):

```python
import opennve

opennve.ensure_driver("SE1-S4")  # WinUSB for connected SE1-S4 cameras without a driver

# or directly when opening a camera
camera = opennve.open_camera(path="cameras.json", name="left", install_driver=True)
```

From the command line: `opennve driver` shows the state, `opennve driver --install --model SE1-S4` installs. See [Python API](api.md#usb-driver).

### Linux

Once after the installation:

```sh
neuromorphic-drivers-install-udev-rules
```

Run the `sudo` command it prints, then reconnect the camera.

### macOS

No driver is required.

## Development installation (from source)

```sh
git clone https://github.com/fr3ak2402/openNVE.git
cd openNVE
pip install neuromorphic_drivers-0.19.2-...whl   # wheel from the neuromorphic-drivers fork
pip install -e ".[test]"
pytest
```
