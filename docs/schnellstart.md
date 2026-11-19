# Schnellstart

## 1. Kamera finden

```sh
opennve list
```

```text
SE1-S4     05110634   USB 3.0 SuperSpeed (5.0 Gb/s)    ok
```

Bleibt die Liste leer, hilft die [Fehlerbehebung](fehlerbehebung.md#kamera-wird-nicht-gefunden).

## 2. Konfiguration erstellen

```sh
opennve template SE1-S4 --serial 05110634 --name left > cameras.json
```

Die Datei enthält alle Einstellungen mit Standardwerten. Für den Anfang reicht eine kurze Version, denn fehlende Felder behalten ihren Standardwert:

```json
{
    "name": "left",
    "model": "SE1-S4",
    "serial": "05110634",
    "settings": {
        "biases": {"diff_on": 20, "diff_off": 20}
    },
    "filter": {
        "background_activity": {"enabled": true}
    }
}
```

```sh
opennve check cameras.json
```

## 3. Events lesen

```python
import opennve

with opennve.open_camera(path="cameras.json") as camera:
    print(camera.info())
    for packet in camera.events():
        print(packet.camera, len(packet.events))
        print(packet.events[:5])  # Felder x, y, p, t
```

## 4. Frames erzeugen (40 ms)

```python
import cv2
import opennve

with opennve.open_camera(path="cameras.json") as camera:
    for window in camera.windows():  # Standard: 40-ms-Fenster
        frame = window.to_frame("color")  # ON grün, OFF rot
        cv2.imshow("SE1-S4", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
cv2.destroyAllWindows()
```

## 5. Beispiele aus dem Repository

```sh
python examples/list_cameras.py                         # Kameras und fertige Konfigurationen
python examples/viewer.py examples/cameras.json          # Live-Ansicht aller Kameras, F = Filter an/aus
python examples/frames.py examples/cameras.json left     # ein Frame pro Zeitfenster, M = Darstellung
```

Weiter mit der [Konfiguration](konfiguration.md) oder der [Python-API](api.md).
