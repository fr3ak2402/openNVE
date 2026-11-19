# Python-API

```python
import opennve
```

Alle hier beschriebenen Namen sind direkt unter `opennve.` erreichbar.

- [Kameras finden](#kameras-finden)
- [Konfiguration](#konfiguration)
- [Kameras öffnen](#kameras-öffnen)
- [EventCamera](#eventcamera)
- [CameraPacket](#camerapacket)
- [EventWindow](#eventwindow)
- [Filter, Ausgabe, Fenster](#filter-ausgabe-fenster)
- [Modelle](#modelle)
- [Hilfsfunktionen](#hilfsfunktionen)

---

## Kameras finden

### `list_cameras(configs=()) -> list[CameraInfo]`

Listet die angeschlossenen Kameras, sortiert nach Modell und Seriennummer. Werden Konfigurationen übergeben, bekommt jede Kamera den Namen der Konfiguration mit gleichem Modell und gleicher Seriennummer.

```python
for camera in opennve.list_cameras(opennve.load_configs("cameras.json")):
    print(camera.name, camera.model, camera.serial, camera.available, camera.error)
```

Wirft `RuntimeError`, wenn das System keinen USB-Zugriff erlaubt.

### `list_cameras_json(configs=(), indent=4) -> str`

Wie `list_cameras`, als JSON-Array:

```json
[
    {
        "model": "SE1-S4",
        "device_name": "SENSING SE1-S4-USB",
        "serial": "05110634",
        "connection": "USB 3.0 SuperSpeed (5.0 Gb/s)",
        "address": "1:8",
        "error": null,
        "name": "left",
        "available": true
    }
]
```

### `CameraInfo`

| Attribut / Methode | Bedeutung |
|---|---|
| `model` | Modell, z. B. `"SE1-S4"` |
| `device_name` | Gerätename, z. B. `"SENSING SE1-S4-USB"` |
| `serial` | Seriennummer oder `None`, wenn die Kamera nicht gelesen werden konnte |
| `connection` | USB-Verbindung, z. B. `"USB 3.0 SuperSpeed (5.0 Gb/s)"` |
| `address` | Bus-Adresse |
| `error` | Fehlermeldung (z. B. fehlende Rechte) oder `None` |
| `name` | Name aus der passenden Konfiguration oder `None` |
| `available` | `True`, wenn die Kamera benutzt werden kann |
| `to_dict()` | Als Dictionary (JSON-kompatibel) |
| `default_config()` | `CameraConfig` mit Standardwerten für diese Kamera |

---

## Konfiguration

Das JSON-Format beschreibt [Konfiguration](konfiguration.md) im Detail.

### `CameraConfig`

```python
config = opennve.CameraConfig(model="SE1-S4", serial="05110634", name="left")
config = opennve.CameraConfig.from_dict({...})
config = opennve.CameraConfig.from_json('{"model": "SE1-S4"}')
```

| Attribut | Typ | Bedeutung |
|---|---|---|
| `name` | `str` | Name der Kamera |
| `model` | `CameraModel` | Modell (`config.model.id` = `"SE1-S4"`) |
| `serial` | `str \| None` | Seriennummer |
| `settings` | `neuromorphic_drivers`-`Configuration` | Hardware-Einstellungen (Dataclass des Modells) |
| `filter` | `FilterConfig` | Software-Filter |
| `output` | `OutputConfig` | Ausgabeformat |
| `window` | `WindowConfig \| None` | Zeitfenster (Standard 40 ms) |

| Methode | Bedeutung |
|---|---|
| `from_dict(data)` / `from_json(text)` | Aus Dictionary / JSON-Text, mit Prüfung aller Werte (`ValueError`) |
| `to_dict()` / `to_json(indent=4)` | Als Dictionary / JSON-Text |
| `copy(**changes)` | Kopie mit Änderungen, z. B. `config.copy(name="right", serial="05110636")` |

### Dateien

| Funktion | Bedeutung |
|---|---|
| `load_configs(path) -> list[CameraConfig]` | Datei mit einer Kamera, einer Liste oder `{"cameras": [...]}` laden |
| `load_config(path, name=None) -> CameraConfig` | Eine Kamera laden, bei mehreren über `name` |
| `save_configs(path, configs)` | Eine `CameraConfig` (als Objekt) oder mehrere (als `{"cameras": [...]}`) speichern |

---

## Kameras öffnen

### `open_camera(config=None, *, path=None, name=None, iterator_timeout=0.02) -> EventCamera`

Öffnet eine Kamera aus einer `CameraConfig`, einem Dictionary, einem JSON-Text oder einer Datei.

```python
camera = opennve.open_camera({"model": "SE1-S4", "serial": "05110634"})
camera = opennve.open_camera(path="cameras.json", name="left")

with opennve.open_camera(path="cameras.json", name="left") as camera:
    ...  # wird am Ende automatisch geschlossen
```

`iterator_timeout` (Sekunden) ist die maximale Wartezeit beim Lesen. Danach liefert `read()` `None`, statt zu blockieren.

### `open_cameras(configs, *, iterator_timeout=0.02, secondary_start_delay=1.0) -> dict[str, EventCamera]`

Öffnet mehrere Kameras und gibt `{name: camera}` zurück. Kameras mit `"clock": "external"` werden zuerst geöffnet, dann nach `secondary_start_delay` Sekunden die übrigen. Schlägt eine Kamera fehl, werden die bereits geöffneten wieder geschlossen. Siehe [Mehrere Kameras](mehrere-kameras.md).

---

## EventCamera

```python
camera = opennve.EventCamera(model="SE1-S4", serial="05110634", name="left",
                             configuration=None, filter=None, output=None, window=None,
                             iterator_timeout=0.02)
camera = opennve.EventCamera.from_config(config)
```

Die Kamera wird beim Erzeugen geöffnet. Eine Kamera darf nur von **einem** Thread gleichzeitig gelesen werden.

Kurzformen mit festem Modell: `SE1S4Driver(serial=None, **kwargs)`, `EVK4Driver(...)`, `EVK3HDDriver(...)`.

### Eigenschaften

| Eigenschaft | Bedeutung |
|---|---|
| `name` | Name aus der Konfiguration |
| `model` | Modell, z. B. `"SE1-S4"` |
| `serial` | Seriennummer der geöffneten Kamera |
| `resolution` | `(Breite, Höhe)`, z. B. `(1280, 720)` |
| `config` | Aktuelle `CameraConfig`. Die Seriennummer wird beim Öffnen eingetragen |
| `configuration` | Aktuelle Hardware-Einstellungen (`config.settings`) |
| `is_open` | `True`, solange die Kamera geöffnet ist |
| `device` | Das zugrunde liegende `neuromorphic_drivers`-Gerät (für Spezialfälle) |

### Lesen

| Methode | Bedeutung |
|---|---|
| `events()` | Generator: liefert fortlaufend `CameraPacket` (gefiltert). Zeitüberschreitungen werden übersprungen |
| `read()` | Ein Paket lesen: `CameraPacket` oder `None` bei Zeitüberschreitung |
| `windows(duration_ms=None)` | Generator: liefert fortlaufend fertige `EventWindow` (Standard aus `window`, sonst 40 ms) |
| `read_windows(duration_ms=None)` | Ein Paket lesen, die dadurch fertig gewordenen Fenster zurückgeben (oft `[]`) |
| `packets()` / `read_packet()` | Ungefilterte Rohdaten `(status, packet)` von `neuromorphic_drivers` |

### Information

| Methode | Bedeutung |
|---|---|
| `info()` | Konfiguration, Auflösung und erlaubte Bias-Bereiche als Dictionary |
| `biases_bounds()` | `{"diff_on": {"minimum": -80, "maximum": 145}, ...}` oder `None` |

### Ändern im laufenden Betrieb

| Methode | Bedeutung |
|---|---|
| `update_configuration(settings)` | Hardware-Einstellungen ändern. Dictionary (nur die angegebenen Felder ändern sich) oder `Configuration` des Modells |
| `update_filter(filter)` | Software-Filter ersetzen (`FilterConfig` oder Dictionary) |
| `update_output(output)` | Ausgabeformat ersetzen, z. B. `{"fields": ["t", "x", "y"]}` |
| `update_window(window)` | Fensterlänge ändern, z. B. `{"duration_ms": 20}`, oder `None`. Das laufende Fenster beginnt neu |
| `apply_config(config)` | Alles auf einmal (Einstellungen, Filter, Ausgabe, Fenster, Name). Modell und Seriennummer müssen passen |

```python
camera.update_configuration({"biases": {"diff_on": 30, "diff_off": 30}})
camera.update_filter({"background_activity": {"enabled": True, "time_us": 2000}})
```

### Schließen

`close()` oder `with`-Block. Danach ist `is_open` `False`.

---

## CameraPacket

Ein Paket gefilterter Events einer Kamera, von `camera.events()` oder `camera.read()`.

| Attribut / Methode | Bedeutung |
|---|---|
| `camera` | Kameraname (einmal pro Paket) |
| `events` | Events im eingestellten Ausgabeformat: NumPy-Array mit den Feldern aus `output.fields`, z. B. `x, y, p, t` |
| `rows()` | Events als Liste von Tupeln: `[(412, 233, 1, 1003), ...]` |
| `to_csv(header=False, delimiter=",", camera=False)` | Events als CSV-Text. `camera=True` schreibt `# camera: <name>` als erste Zeile |
| `polarity_events` | Gefilterte Events im Originalformat (`t, x, y, on`) oder `None` |
| `trigger_events` | Trigger-Events (externer Eingang) oder `None` |
| `raw_event_count` | Anzahl Events **vor** dem Filter |
| `status` | Status von `neuromorphic_drivers`, z. B. `status.ring.backlog` (Rückstand) und `status.ring.current_t` (Kamerazeit) |
| `packet` | Das ursprüngliche Paket von `neuromorphic_drivers` |

```python
for packet in camera.events():
    events = packet.events
    on = events[events["p"] == 1]
    print(packet.camera, len(events), events["t"][-1] if len(events) else None)
```

`rows()` und `to_csv()` erzeugen ein Python-Objekt pro Event. Für hohe Event-Raten besser direkt mit dem NumPy-Array `events` arbeiten.

---

## EventWindow

Die Events eines Zeitfensters, von `camera.windows()` oder `camera.read_windows()`. Siehe [Zeitfenster und Frames](zeitfenster.md).

| Attribut / Methode | Bedeutung |
|---|---|
| `camera` | Kameraname |
| `start_t`, `end_t` | Fenstergrenzen in µs Kamerazeit (`start_t` gehört dazu, `end_t` nicht) |
| `duration_ms` | Fensterlänge in ms |
| `width`, `height` | Sensorgröße |
| `len(window)` | Anzahl Events |
| `events` | Events im Ausgabeformat (wie `CameraPacket.events`) |
| `polarity_events` | Events im Originalformat (`t, x, y, on`) |
| `rows()`, `to_csv(...)` | Wie bei `CameraPacket` |
| `to_frame(mode="count")` | Bild (Höhe × Breite) aus den Events: `count`, `polarity`, `binary` oder `color` |

---

## Filter, Ausgabe, Fenster

| Klasse | Felder | Bedeutung |
|---|---|---|
| `FilterConfig` | `polarity`, `background_activity` | Software-Filter. `from_dict`, `to_dict`, `is_active` |
| `BackgroundActivityConfig` | `enabled`, `time_us`, `min_neighbors` | Rauschfilter-Einstellungen |
| `OutputConfig` | `fields` | Ausgabeformat. `format(events)` wandelt `t, x, y, on` in das Ausgabeformat um, `dtype()` liefert den NumPy-Typ |
| `WindowConfig` | `duration_ms` | Fensterlänge (Standard 40). `duration_us` in Mikrosekunden |

Für eigene Verarbeitungsketten ohne `EventCamera`:

| Klasse | Bedeutung |
|---|---|
| `EventFilter(config, width, height)` | Wendet eine `FilterConfig` an: `filtered = event_filter(events)` |
| `BackgroundActivityFilter(width, height, time_us, min_neighbors=1)` | Nur der Rauschfilter |
| `WindowAccumulator(duration_us, camera, width, height, output_config=None)` | Teilt Events in Fenster: `push(events, current_t=None) -> list[EventWindow]`, `reset()` |

```python
accumulator = opennve.WindowAccumulator(40_000, "left", 1280, 720)
for packet in camera.events():
    for window in accumulator.push(packet.polarity_events, packet.status.ring.current_t):
        ...
```

---

## Modelle

| Name | Bedeutung |
|---|---|
| `MODELS` | Alle unterstützten Modelle (`CameraModel`) |
| `get_model(name) -> CameraModel` | Modell aus Kurzname (`"SE1-S4"`), Gerätename oder Modulname. `ValueError`, wenn unbekannt |
| `CameraModel.id` | Kurzname, z. B. `"SE1-S4"` |
| `CameraModel.name` | Gerätename, z. B. `"SENSING SE1-S4-USB"` |
| `CameraModel.module` | `neuromorphic_drivers`-Modul, z. B. `nd.sensing_se1_s4` |
| `CameraModel.configuration_type` | Dataclass der Hardware-Einstellungen |

---

## Hilfsfunktionen

| Funktion | Bedeutung |
|---|---|
| `to_rows(events)` | Strukturiertes NumPy-Array → Liste von Tupeln |
| `to_csv(events, header=False, delimiter=",", camera=None)` | Strukturiertes NumPy-Array → CSV-Text |
| `settings_from_dict(model, data)` | JSON-Einstellungen → `Configuration` des Modells |
| `settings_to_dict(settings)` | `Configuration` → JSON-Einstellungen |
| `__version__` | openNVE-Version |
| `NEUROMORPHIC_DRIVERS_VERSION` | Version des eingebetteten Treibers, z. B. `"0.19.2"` |

## Fehler

| Ausnahme | Wann |
|---|---|
| `ValueError` | Ungültige Konfiguration (die Meldung nennt das Feld), unbekanntes Modell, Konfiguration passt nicht zur Kamera |
| `TypeError` | `Configuration` eines anderen Modells übergeben |
| `RuntimeError` | Kamera geschlossen, nicht gefunden oder USB-Fehler (von `neuromorphic_drivers`) |
