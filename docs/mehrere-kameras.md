# Mehrere Kameras

## Kameras unterscheiden

Jede Kamera hat eine Seriennummer (`opennve list`). In der Konfiguration bekommt jede Kamera über `serial` einen festen Namen:

```json
{
    "cameras": [
        {"name": "left",  "model": "SE1-S4", "serial": "05110634"},
        {"name": "right", "model": "SE1-S4", "serial": "05110636"}
    ]
}
```

Tipp: Steck die Kameras einzeln an, notiere die Seriennummer und kleb sie aufs Gehäuse.

## Öffnen und lesen

```python
import numpy as np
import opennve

cameras = opennve.open_cameras(opennve.load_configs("cameras.json"))
try:
    names = list(cameras)
    backlogs = np.zeros(len(names), dtype=np.int64)
    while True:
        # Die Kamera mit dem größten Rückstand zuerst lesen
        index = int(np.argmax(backlogs))
        packet = cameras[names[index]].read()
        backlogs += 1
        if packet is None:
            backlogs[index] = 0
            continue
        backlogs[index] = packet.status.ring.backlog if packet.status.ring else 0
        print(packet.camera, len(packet.events))
finally:
    for camera in cameras.values():
        camera.close()
```

Die Kameras werden abwechselnd aus einem Thread gelesen. Durch die Auswahl nach Rückstand (*backlog*) läuft keine Kamera voll, während die andere gelesen wird. `examples/viewer.py` zeigt das vollständig, mit Anzeige aller Kameras nebeneinander.

Für Zeitfenster gibt es `read_windows()`, siehe [Zeitfenster und Frames](zeitfenster.md#mehrere-kameras).

## Uhren synchronisieren

Ohne Synchronisation hat jede Kamera ihre eigene Uhr. Die Zeitstempel zweier Kameras passen dann nicht exakt zusammen und laufen mit der Zeit auseinander. Mit einem Sync-Kabel zwischen den Kameras gibt eine Kamera (*Primary*) ihren Takt an die anderen (*Secondary*) weiter:

```json
{
    "cameras": [
        {"name": "left",  "model": "SE1-S4", "serial": "05110634",
         "settings": {"clock": "internal_with_output_enabled"}},
        {"name": "right", "model": "SE1-S4", "serial": "05110636",
         "settings": {"clock": "external"}}
    ]
}
```

| `clock` | Rolle |
|---|---|
| `internal` | eigene Uhr, keine Synchronisation (Standard) |
| `internal_with_output_enabled` | **Primary**: eigene Uhr, Takt am Sync-Ausgang |
| `external` | **Secondary**: Takt vom Sync-Eingang |

Die Secondaries müssen bereit sein, bevor die Primary ihren Takt startet. `open_cameras` öffnet deshalb automatisch zuerst alle Kameras mit `"clock": "external"`, wartet `secondary_start_delay` Sekunden (Standard 1) und öffnet dann die übrigen.

Mit synchronisierten Uhren haben die [Zeitfenster](zeitfenster.md) aller Kameras dieselben Grenzen. Fenster mit gleichem `start_t` gehören zusammen.

> **Ungeprüft bei der SE1-S4:** Die Pinbelegung für Sync-Ein- und -Ausgang hat der Treiber vom Prophesee EVK4 übernommen. Prüfe die Synchronisation einmal: Bei korrekter Verbindung bleibt der Zeitversatz zwischen den Kameras über Minuten konstant, statt langsam zu wachsen.

## USB und Stromversorgung

- Jede SE1-S4 kann bis zu 1,6 Gbit/s liefern. An einem gemeinsamen USB-3-Hub (5 Gbit/s) sind zwei Kameras unkritisch.
- **Raspberry Pi 5:** Je eine Kamera an jeden der beiden blauen USB-3-Ports. Mit dem 27-W-Netzteil liefert der Pi insgesamt höchstens 1,6 A über USB. Für zwei Kameras ist ein aktiver USB-3-Hub mit eigenem Netzteil sicherer.
