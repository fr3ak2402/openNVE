# Entwicklung

## Projektstruktur

```text
opennve/
├── __init__.py      öffentliche API
├── __main__.py      Kommandozeile (opennve ...)
├── camera.py        EventCamera, CameraPacket, open_camera, open_cameras
├── config.py        CameraConfig, JSON laden/speichern, Prüfung der Hardware-Einstellungen
├── devices.py       SE1S4Driver, EVK4Driver, EVK3HDDriver
├── discovery.py     list_cameras, CameraInfo
├── filters.py       FilterConfig, Rauschfilter, Polaritätsfilter
├── models.py        unterstützte Modelle (aus neuromorphic_drivers)
├── output.py        OutputConfig (Felder x, y, p, t), CSV
└── window.py        WindowConfig, WindowAccumulator, EventWindow, Frames
examples/            cameras.json, viewer.py, frames.py, list_cameras.py
tests/               pytest, ohne Hardware (Kamera wird simuliert)
tools/bundle_wheel.py  bettet neuromorphic_drivers in die Wheels ein
docs/                diese Dokumentation
```

## Tests

```sh
pip install -e ".[test]"
pytest
```

Die Tests brauchen keine Kamera: `neuromorphic_drivers.open` und `list_devices` werden durch Attrappen ersetzt. Getestet werden u. a. Konfiguration und Prüfung, Filter, Ausgabeformat, Zeitfenster, Frames, Kameraklasse und Kommandozeile.

## Hardware-Einstellungen neuer Modelle

`settings` wird nicht von Hand gepflegt. `config.py` liest die Felder und Typen direkt aus den `Configuration`-Dataclasses von `neuromorphic_drivers` (verschachtelte Dataclasses, Enums, `Optional`, Zahlentypen mit Wertebereich). Neue Modelle oder Felder im Treiber sind dadurch automatisch verfügbar.

## Wheels bauen (GitHub Actions)

`.github/workflows/build.yml` läuft bei jedem Push auf `main`, bei Pull Requests und von Hand (*Actions → Build → Run workflow*):

1. `neuromorphic_drivers` wird aus [fr3ak2402/neuromorphic-drivers](https://github.com/fr3ak2402/neuromorphic-drivers) gebaut, für Python 3.14 und 3.15 auf Windows x64, Linux x64/ARM64 und macOS ARM/Intel.
2. `tools/bundle_wheel.py` fügt jedes Treiber-Wheel mit dem Paket `opennve` zu einem Wheel zusammen: gleiche Plattform-Tags, Konsolenbefehle beider Pakete, Lizenzen beider Pakete.
3. Jedes Wheel wird in einer frischen Umgebung installiert und mit allen Tests geprüft (`.github/actions/test-wheel`).
4. Die Wheels liegen als Artifacts am Lauf. Bei Tags `v*` werden sie zusätzlich an ein GitHub-Release gehängt.

Welcher Stand des Treibers eingebettet wird, legt `NEUROMORPHIC_DRIVERS_REF` im Workflow fest (Standard `main`). Für reproduzierbare Releases einen Tag oder Commit eintragen.

Lokal bündeln:

```sh
python tools/bundle_wheel.py pfad/zu/neuromorphic_drivers-wheels/ --out dist
```

## Release

1. Version in `pyproject.toml` erhöhen (`version = "0.1.1"`).
2. Committen und pushen.
3. Tag setzen:
   ```sh
   git tag v0.1.1
   git push --tags
   ```
4. Die Action baut alle Wheels und hängt sie an das Release `v0.1.1`.

## Neue Python-Version

Der Treiber ist in Rust geschrieben und wird für jede Python-Version einzeln kompiliert. Für eine neue Version, z. B. 3.16:

- in `build.yml` bei `interpreters` (Linux) und `python-version` ergänzen,
- einen Testschritt `Test (Python 3.16)` hinzufügen.

## Lizenzen

`neuromorphic_drivers` steht unter der MIT-Lizenz. Die Lizenz liegt in jedem Wheel unter `opennve-*.dist-info/licenses/neuromorphic_drivers/LICENSE`. Der WinUSB-Installer (`winusb_installer.exe`, LGPL-3.0) wird nicht mitgeliefert, sondern erst bei Bedarf heruntergeladen.
