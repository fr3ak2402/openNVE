# openNVE – Dokumentation

openNVE (Open Neuromorphic Vision Engine) ist eine Python-Bibliothek für Eventkameras. Sie findet angeschlossene Kameras, konfiguriert sie über JSON, liest gefilterte Events und fasst sie zu Zeitfenstern und Frames zusammen.

| Seite | Inhalt |
|---|---|
| [Installation](installation.md) | Wheel installieren, Kameratreiber unter Windows und Linux |
| [Schnellstart](schnellstart.md) | In fünf Minuten von der angeschlossenen Kamera zum ersten Frame |
| [Konfiguration (JSON)](konfiguration.md) | Alle Felder der Kamera-Konfiguration: Hardware, Filter, Ausgabeformat, Zeitfenster |
| [Python-API](api.md) | Referenz aller Funktionen, Klassen und Methoden |
| [Zeitfenster und Frames](zeitfenster.md) | Events in Fenstern sammeln (Standard 40 ms) und Bilder erzeugen |
| [Mehrere Kameras](mehrere-kameras.md) | Mehrere Kameras gleichzeitig lesen und synchronisieren |
| [Kommandozeile](kommandozeile.md) | `opennve list`, `template`, `check`, `models` |
| [Fehlerbehebung](fehlerbehebung.md) | Kamera wird nicht gefunden, Rauschen, hohe Event-Raten |
| [Entwicklung](entwicklung.md) | Tests, Wheel-Build, Releases, eingebetteter Treiber |

## Überblick

```text
           ┌────────────────────────────────────────────────────────────┐
JSON  ───► │ CameraConfig: name, model, serial, settings, filter,       │
           │               output, window                               │
           └──────────────────────────┬─────────────────────────────────┘
                                      ▼
Kamera ──► EventCamera ──► Filter ──► CameraPacket  (events im Format x, y, p, t)
 (USB)     (neuromorphic_drivers)     └──► EventWindow (40 ms) ──► to_frame()
```

- **Kameras finden:** `opennve.list_cameras()`
- **Konfigurieren:** eine JSON-Datei pro Aufbau, mit einem Eintrag je Kamera
- **Lesen:** `camera.events()` liefert Pakete mit gefilterten Events, `camera.windows()` liefert Zeitfenster
- **Frames:** `window.to_frame("count" | "polarity" | "binary" | "color")`

## Unterstützte Kameras

| Modell (`model`) | Gerät |
|---|---|
| `SE1-S4` | SENSING SE1-S4-USB (Sony IMX646, 1280 × 720) |
| `EVK4` | Prophesee EVK4 (Sony IMX636, 1280 × 720) |
| `EVK3-HD` | Prophesee EVK3 HD |
| `EVK1-GEN4` | Prophesee EVK1 Gen4 |
| `DAVIS346` | iniVation DAVIS 346 |
| `DVXPLORER` | iniVation DVXplorer |
| `CENTURYARKS-VGA` | CenturyArks VGA |
| `TRITON` | Lucid Triton |

Die Hardware-Anbindung übernimmt [neuromorphic_drivers](https://github.com/fr3ak2402/neuromorphic-drivers) (Fork mit SE1-S4-Treiber). Er ist in den openNVE-Wheels bereits enthalten.
