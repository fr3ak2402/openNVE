# Installation

## Voraussetzungen

- Python **3.14 oder neuer**
- Windows x64, Linux x64, Linux ARM64 (z. B. Raspberry Pi 5 mit 64-Bit-System) oder macOS
- Eine Eventkamera an einem **USB-3-Port** (bei der SE1-S4 bis 1,6 Gbit/s)

## Wheel installieren

Lade das passende Wheel aus den [Releases](https://github.com/fr3ak2402/openNVE/releases) oder aus den Artifacts der GitHub Action *Build* herunter.

| Dateiname enthält | Plattform |
|---|---|
| `win_amd64` | Windows x64 |
| `manylinux_2_28_x86_64` | Linux x64 |
| `manylinux_2_28_aarch64` | Linux ARM64 (Raspberry Pi 5) |
| `macosx_..._arm64` | macOS Apple Silicon |
| `macosx_..._x86_64` | macOS Intel |

`cp314` steht für Python 3.14, `cp315` für Python 3.15. Deine Version zeigt `python --version`.

```sh
pip uninstall -y neuromorphic_drivers     # nur falls separat installiert (siehe unten)
pip install opennve-0.1.0-cp314-cp314-win_amd64.whl
```

pip installiert numpy und OpenCV automatisch mit.

> **Wichtig:** Das Wheel enthält `neuromorphic_drivers` 0.19.2 mit dem SE1-S4-Treiber. Ist zusätzlich ein separates `neuromorphic_drivers` installiert, überschreiben sich die beiden Pakete gegenseitig. Deshalb vorher deinstallieren.

> **Hinweis:** Das Wheel immer **ohne** `-r` installieren. `pip install -r datei.whl` liest das Wheel als Textdatei und bricht mit `UnicodeDecodeError` ab.

Prüfen:

```sh
python -c "import opennve; print(opennve.__version__, opennve.NEUROMORPHIC_DRIVERS_VERSION)"
opennve models
```

## Kameratreiber

### Windows

Windows erlaubt Programmen den Zugriff auf ein USB-Gerät nur, wenn es einen Treiber hat. openNVE nutzt den Microsoft-Treiber **WinUSB**. Einmal pro PC und Kameramodell, mit angeschlossener Kamera:

```powershell
neuromorphic-drivers-install-windows-driver
```

Bestätige die Administrator-Abfrage. Die Installation kann einige Minuten dauern. Der Befehl macht dasselbe wie [Zadig](https://zadig.akeo.ie).

```powershell
neuromorphic-drivers-install-windows-driver --list   # Kameras und aktuellen Treiber anzeigen
neuromorphic-drivers-install-windows-driver --force  # vorhandenen (Hersteller-)Treiber ersetzen
```

Kameras mit einem anderen Treiber, z. B. von Metavision, werden ohne `--force` nicht verändert. Mit WinUSB funktioniert die Hersteller-Software unter Umständen nicht mehr. Zurück geht es im Geräte-Manager unter *Treiber → Vorheriger Treiber*.

Automatisch beim Programmstart:

```python
import sys

if sys.platform == "win32":
    from neuromorphic_drivers.windows_driver import install_windows_driver, list_windows_devices

    if any(not device.has_driver for device in list_windows_devices()):
        install_windows_driver()
```

### Linux

Einmal nach der Installation:

```sh
neuromorphic-drivers-install-udev-rules
```

Den angezeigten `sudo`-Befehl ausführen und die Kamera neu einstecken.

### macOS

Kein Treiber nötig.

## Entwicklungsinstallation (aus dem Quellcode)

```sh
git clone https://github.com/fr3ak2402/openNVE.git
cd openNVE
pip install neuromorphic_drivers-0.19.2-...whl   # Wheel aus dem neuromorphic-drivers-Fork
pip install -e ".[test]"
pytest
```
