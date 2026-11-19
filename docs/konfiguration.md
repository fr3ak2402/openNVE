# Konfiguration (JSON)

Jede Kamera wird durch ein JSON-Objekt beschrieben:

```json
{
    "name": "left",
    "model": "SE1-S4",
    "serial": "05110634",
    "settings": {
        "biases": {"diff_on": 20, "diff_off": 20},
        "clock": "internal",
        "rate_limiter": null,
        "enable_external_trigger": true,
        "enable_output": true
    },
    "filter": {
        "polarity": "both",
        "background_activity": {"enabled": true, "time_us": 5000, "min_neighbors": 1}
    },
    "output": {
        "fields": ["x", "y", "p", "t"]
    },
    "window": {
        "duration_ms": 40
    }
}
```

Nur `model` ist Pflicht. Fehlende Felder behalten ihren Standardwert, auch innerhalb von Abschnitten: `"biases": {"diff_on": 20}` ändert nur `diff_on`. Unbekannte Felder und ungültige Werte werden mit einer Fehlermeldung abgelehnt, die das betroffene Feld nennt, z. B. `settings.biases.diff_on must be in [-32768, 32767]`.

## Dateiformate

Eine Datei enthält entweder

- **eine Kamera** (ein Objekt wie oben),
- **eine Liste** von Kameras (`[{...}, {...}]`) oder
- **ein Objekt mit `cameras`** (`{"cameras": [{...}, {...}]}`). So speichert `save_configs` mehrere Kameras.

Die Namen müssen innerhalb einer Datei eindeutig sein. Prüfen mit `opennve check datei.json`.

## Allgemeine Felder

| Feld | Typ | Standard | Bedeutung |
|---|---|---|---|
| `name` | Text | `<model>-<serial>` | Name der Kamera in deinem Aufbau, z. B. `left`. Steht in jedem Paket als `packet.camera` |
| `model` | Text | – (Pflicht) | Kameramodell, siehe `opennve models`. Groß-/Kleinschreibung egal, auch `se1_s4` oder `SENSING SE1-S4-USB` |
| `serial` | Text oder `null` | `null` | Seriennummer **als Text** (`"05110634"`, nicht `5110634`). `null` öffnet die erste Kamera dieses Modells |

## `settings` – Hardware

Diese Einstellungen gehen direkt an die Kamera. Welche Felder es gibt, hängt vom Modell ab. `opennve template <modell>` zeigt alle mit Standardwerten.

### SE1-S4 und EVK4

| Feld | Typ | Standard | Bedeutung |
|---|---|---|---|
| `biases` | Objekt | alle `0` | Empfindlichkeit des Sensors (siehe unten) |
| `clock` | Text | `"internal"` | `internal`: eigene Uhr. `internal_with_output_enabled`: eigene Uhr und Takt am Sync-Ausgang (*Primary*). `external`: Takt vom Sync-Eingang (*Secondary*). Siehe [Mehrere Kameras](mehrere-kameras.md) |
| `rate_limiter` | Objekt oder `null` | `null` | Begrenzt die Event-Rate im Sensor: `{"reference_period_us": 200, "maximum_events_per_period": 4000}` erlaubt höchstens 4000 Events pro 200 µs (= 20 Mio. Events/s). Überzählige Events werden zufällig verworfen |
| `enable_external_trigger` | `true`/`false` | `true` | Trigger-Eingang aktiv, Signale kommen als `packet.trigger_events` |
| `enable_output` | `true`/`false` | `true` | Event-Ausgabe des Sensors |
| `x_mask`, `y_mask`, `pixel_mask`, `mask_intersection_only` | Listen / `true`/`false` | keine Maske | Spalten, Zeilen oder einzelne Pixel abschalten. Bit-Masken, am einfachsten in Python erzeugen (siehe unten). Werden nur gespeichert, wenn sie vom Standard abweichen |

#### Biases (SE1-S4)

Beim SE1-S4 sind die Biases **Abweichungen vom Werkswert** der Kamera: `0` bedeutet Werkseinstellung. Der Treiber begrenzt die Werte auf die für den IMX646 empfohlenen Bereiche (Quelle: OpenEB). Die erlaubten Bereiche liefert `camera.biases_bounds()`.

| Bias | Bereich | Wirkung bei höherem Wert |
|---|---|---|
| `diff_on` | −80 … 145 | weniger empfindlich für **hellere** Stellen, weniger ON-Events |
| `diff_off` | −30 … 200 | weniger empfindlich für **dunklere** Stellen, weniger OFF-Events |
| `hpf` | 0 … 120 | langsame Helligkeitsänderungen und Rauschen werden stärker unterdrückt |
| `fo` | −20 … 0 | höhere Bandbreite. Niedrigere Werte = weniger Rauschen, aber träger |
| `refr` | −20 … 235 | Pixel ist nach einem Event länger gesperrt, weniger Event-Bursts |
| `diff` | −25 … 23 | Referenzpegel. Normalerweise nicht ändern |
| `pr`, `inv`, `reqpuy`, `reqpux`, `sendreqpdy`, `unknown_1`, `unknown_2` | 8-Bit-Bereich um den Werkswert | interne Ströme. Normalerweise nicht ändern |

Empfehlung gegen Rauschen: mit `diff_on` und `diff_off` bei 20–40 beginnen und schrittweise erhöhen. Zu hohe Werte lassen schwache Kanten verschwinden.

Beim EVK3 HD sind die Biases absolute Werte. Die Standardwerte zeigt `opennve template EVK3-HD`.

#### Masken (Region of Interest)

Ein gesetztes Bit schaltet eine Spalte (`x_mask`), eine Zeile (`y_mask`) oder ein Pixel (`pixel_mask`) ab. Mit `"mask_intersection_only": false` ist ein Pixel abgeschaltet, wenn seine Spalte **oder** seine Zeile maskiert ist. Am einfachsten erzeugst du die Masken mit der Hilfsklasse des Treibers. Beispiel: nur ein Rechteck von 480 × 320 Pixeln ab (400, 200) bleibt aktiv:

```python
import neuromorphic_drivers as nd
import opennve

mask = nd.sensing_se1_s4.RowColumnMask(set=True)        # alles abgeschaltet
mask.clear_rectangle(x=400, y=200, width=480, height=320)  # Rechteck wieder aktiv

config = opennve.CameraConfig.from_dict({
    "name": "left",
    "model": "SE1-S4",
    "settings": {
        "x_mask": list(mask.x_mask()),
        "y_mask": list(mask.y_mask()),
        "mask_intersection_only": False,
    },
})
opennve.save_configs("left.json", config)  # die Masken stehen jetzt in der JSON-Datei
```

Für den EVK4 heißt die Klasse `nd.prophesee_evk4.RowColumnMask`. Maskierte Pixel erzeugen gar keine Events, das senkt auch die USB-Last.

### Andere Modelle

| Modell | Felder in `settings` |
|---|---|
| `EVK3-HD` | `biases`, `rate_limiter` |
| `DAVIS346` | `biases`, `region_of_interest`, `activity_filter`, `skip_events_every`, `polarity_filter`, `exposure_us`, `frame_interval_us` |
| übrige | `opennve template <modell>` |

## `filter` – Software-Filter

Die Filter laufen auf dem PC, nachdem die Events gelesen wurden. Sie wirken auf `packet.events` und auf die Zeitfenster.

| Feld | Typ | Standard | Bedeutung |
|---|---|---|---|
| `polarity` | Text | `"both"` | `both`: alle Events. `on`: nur hellere Pixel. `off`: nur dunklere Pixel |
| `background_activity.enabled` | `true`/`false` | `false` | Rauschfilter ein/aus |
| `background_activity.time_us` | Zahl | `5000` | Zeitfenster in µs (1 … 1 000 000). Kleiner = stärker filtern |
| `background_activity.min_neighbors` | Zahl | `1` | Benötigte aktive Nachbarpixel (1 … 8). Größer = stärker filtern |

**Rauschfilter (*Background Activity Filter*):** Ein Event bleibt nur erhalten, wenn mindestens `min_neighbors` der 8 Nachbarpixel innerhalb der letzten `time_us` Mikrosekunden ebenfalls ein Event hatten. Echte Kanten lösen benachbarte Pixel fast gleichzeitig aus, Rauschen und „heiße“ Pixel nicht. In Tests mit künstlichen Daten blieben etwa 10 % des Rauschens übrig, eine bewegte Kante blieb vollständig erhalten. Der Filter schafft etwa 9 Mio. Events/s.

## `output` – Ausgabeformat

| Feld | Typ | Standard | Bedeutung |
|---|---|---|---|
| `fields` | Liste | `["x", "y", "p", "t"]` | Felder der Events, in dieser Reihenfolge. Beliebige Auswahl, jedes Feld höchstens einmal. Auch als Text `"t, x, y"` möglich |

| Feld | Typ | Bedeutung |
|---|---|---|
| `x` | uint16 | Pixelspalte (0 … Breite−1) |
| `y` | uint16 | Pixelzeile (0 … Höhe−1) |
| `p` | uint8 | Polarität: `1` = heller (ON), `0` = dunkler (OFF) |
| `t` | uint64 | Zeitstempel der Kamera in µs |

Der **Kameraname** ist kein Event-Feld. Er steht einmal pro Paket bzw. Fenster in `packet.camera` / `window.camera`. Bei CSV-Ausgabe kann er mit `to_csv(camera=True)` einmal als erste Zeile geschrieben werden.

## `window` – Zeitfenster

| Feld | Typ | Standard | Bedeutung |
|---|---|---|---|
| `duration_ms` | Zahl | `40` | Länge der Zeitfenster in ms (0,01 … 60 000) |

- Ohne `window`-Abschnitt gelten **40 ms**.
- `"window": null` schaltet die Zeitfenster aus. `camera.windows()` braucht dann `duration_ms` als Argument.

Details unter [Zeitfenster und Frames](zeitfenster.md).

## Beispiel: zwei synchronisierte Kameras

```json
{
    "cameras": [
        {
            "name": "left",
            "model": "SE1-S4",
            "serial": "05110634",
            "settings": {"biases": {"diff_on": 20, "diff_off": 20}, "clock": "internal_with_output_enabled"},
            "filter": {"background_activity": {"enabled": true}}
        },
        {
            "name": "right",
            "model": "SE1-S4",
            "serial": "05110636",
            "settings": {"biases": {"diff_on": 20, "diff_off": 20}, "clock": "external"},
            "filter": {"background_activity": {"enabled": true}}
        }
    ]
}
```

## Konfiguration in Python

```python
import opennve

config = opennve.CameraConfig.from_dict({"model": "SE1-S4", "serial": "05110634", "name": "left"})
config = opennve.CameraConfig.from_json(text)
configs = opennve.load_configs("cameras.json")
config = opennve.load_config("cameras.json", "left")

print(config.to_json())
opennve.save_configs("cameras.json", configs)  # mehrere Kameras
opennve.save_configs("left.json", config)      # eine Kamera

right = config.copy(name="right", serial="05110636")
```
