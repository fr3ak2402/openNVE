# Zeitfenster und Frames

Eine Eventkamera liefert keine Bilder, sondern einen fortlaufenden Strom einzelner Events. Für Algorithmen, die Bilder erwarten, z. B. ein neuronales Netz, sammelt openNVE die Events einer festen Zeitspanne und erzeugt daraus ein Frame.

## Fensterlänge einstellen

```json
"window": {"duration_ms": 40}
```

- **Standard: 40 ms** (25 Frames pro Sekunde), auch ohne `window`-Abschnitt
- Erlaubt: 0,01 ms bis 60 000 ms
- `"window": null` schaltet die Fenster aus
- Im Code: `camera.windows(duration_ms=20)` überschreibt die Konfiguration, `camera.update_window({"duration_ms": 20})` ändert sie dauerhaft

## Verwendung

```python
import opennve

with opennve.open_camera(path="cameras.json", name="left") as camera:
    for window in camera.windows():
        print(window.camera, window.start_t, window.end_t, len(window))
        events = window.events            # alle Events des Fensters (x, y, p, t)
        frame = window.to_frame("count")  # Bild 720 × 1280
```

## Wie die Fenster gebildet werden

- **Nach Kamerazeit:** Die Fenster richten sich nach den Zeitstempeln der Kamera, nicht nach der PC-Uhr. Bei 40 ms sind die Grenzen immer Vielfache von 40 ms: [0, 40), [40, 80), … ms Kamerazeit. Zwei [synchronisierte Kameras](mehrere-kameras.md) liefern dadurch Fenster mit exakt denselben Grenzen, ihre Frames passen paarweise zusammen.
- **Vollständig:** Jedes Event landet in genau einem Fenster, auch wenn ein USB-Paket über eine Fenstergrenze reicht. `start_t` gehört zum Fenster, `end_t` nicht.
- **Pünktlich auch ohne Bewegung:** Ein Fenster wird fertig, sobald die Kamerazeit sein Ende überschreitet (`status.ring.current_t`). In ruhigen Phasen entstehen dadurch leere Fenster. Nur sehr lange Pausen (mehr als 10 000 Fenster am Stück) werden übersprungen.
- **Gefiltert:** Die Fenster enthalten die Events nach dem Software-Filter (`filter`).
- **Verzögerung:** Ein Fenster wird ausgegeben, sobald die Kamera Daten nach seinem Ende geschickt hat, also mit wenigen Millisekunden Verzögerung.

## Frames erzeugen

`window.to_frame(mode)` liefert ein NumPy-Array der Größe Höhe × Breite:

| Modus | Datentyp | Inhalt |
|---|---|---|
| `"count"` | `uint16` | Anzahl Events pro Pixel |
| `"polarity"` | `int16` | Anzahl ON-Events minus Anzahl OFF-Events pro Pixel: positiv = heller geworden, negativ = dunkler |
| `"binary"` | `uint8` | `255`, wo mindestens ein Event war, sonst `0` |
| `"color"` | `uint8`, 3 Kanäle | BGR-Bild für OpenCV: ON grün, OFF rot (das letzte Event pro Pixel gewinnt) |

`count` und `polarity` sind für die Weiterverarbeitung gedacht. Für die Anzeige mit OpenCV müssen sie skaliert werden:

```python
import numpy as np

count = window.to_frame("count")
image = np.clip(count.astype(np.float32) * 64, 0, 255).astype(np.uint8)

polarity = window.to_frame("polarity")
image = np.clip(polarity.astype(np.float32) * 64 + 128, 0, 255).astype(np.uint8)  # grau = keine Änderung
```

Eigene Darstellungen lassen sich direkt aus `window.polarity_events` bauen, z. B. eine *Time Surface* aus den Zeitstempeln `t`.

## Mehrere Kameras

`windows()` blockiert, bis ein Fenster fertig ist. Für mehrere Kameras in einer Schleife gibt es `read_windows()`: Es liest ein Paket und gibt die dadurch fertig gewordenen Fenster zurück, oft eine leere Liste.

```python
cameras = opennve.open_cameras(opennve.load_configs("cameras.json"))
latest = {}
while True:
    for name, camera in cameras.items():
        for window in camera.read_windows():
            latest[name] = window  # Fenster mit gleichem start_t gehören zusammen
    # z. B.: wenn latest["left"].start_t == latest["right"].start_t -> Stereo-Paar verarbeiten
```

## Geschwindigkeit

Gemessen mit 2 Millionen Events (1280 × 720):

| Schritt | Dauer |
|---|---|
| In Fenster aufteilen | ca. 16 ms |
| `to_frame("binary")` | ca. 22 ms |
| `to_frame("count")` | ca. 33 ms |
| `to_frame("polarity")` | ca. 59 ms |
| `to_frame("color")` | ca. 130 ms |

Bei 40-ms-Fenstern entsprechen 2 Millionen Events pro Fenster 50 Mio. Events/s. Bei realistischen Raten bleibt also reichlich Zeit für die Weiterverarbeitung.

## Ohne EventCamera

Für eigene Datenquellen, z. B. aufgezeichnete Events:

```python
accumulator = opennve.WindowAccumulator(duration_us=40_000, camera="left", width=1280, height=720)
for chunk in chunks:  # NumPy-Arrays mit Feldern t, x, y, on, nach t sortiert
    for window in accumulator.push(chunk):
        frame = window.to_frame("count")
```
