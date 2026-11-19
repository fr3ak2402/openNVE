# Fehlerbehebung

## Kamera wird nicht gefunden

`opennve list` zeigt `No camera found` oder `ndls` eine leere Tabelle.

1. **Erkennt das Betriebssystem die Kamera?**
   - Windows: Geräte-Manager → Kamera → *Eigenschaften* → *Details* → *Hardware-IDs*. Für die SE1-S4 muss dort `USB\VID_04B4&PID_00C4` stehen.
   - Linux: `lsusb | grep 04b4`
   - Taucht die Kamera nicht auf, liegt es an Kabel, Port oder Stromversorgung.
2. **Windows: Ist ein Treiber installiert?**
   ```powershell
   neuromorphic-drivers-install-windows-driver --list
   ```
   Steht dort `driver: (none)` oder ein anderer Treiber als `WinUSB`, installierst du WinUSB mit `neuromorphic-drivers-install-windows-driver`. Ohne Treiber ist die Kamera für Programme unsichtbar.
3. **Läuft Python in WSL, Docker oder einer VM?** Dort ist USB standardmäßig nicht durchgereicht (unter WSL z. B. `usbipd attach`).

## Kamera wird gefunden, aber mit Fehler

- **Linux, Zugriff verweigert:** `neuromorphic-drivers-install-udev-rules` ausführen, den angezeigten `sudo`-Befehl ausführen und die Kamera neu einstecken.
- **`unsupported sensor`:** Der Sensor meldet eine unerwartete ID. Bitte mit der Ausgabe als Issue melden.

## `pip install` bricht mit `UnicodeDecodeError` ab

Das Wheel wurde mit `-r` installiert. Richtig ist es ohne `-r`:

```sh
pip install opennve-0.1.0-cp314-cp314-win_amd64.whl
```

## `No module named neuromorphic_drivers.sensing_se1_s4` oder falsche Treiberversion

Ein separat installiertes `neuromorphic_drivers`, z. B. von PyPI, überschreibt die eingebettete Version:

```sh
pip uninstall -y neuromorphic_drivers opennve
pip install opennve-...whl
python -c "import opennve; print(opennve.NEUROMORPHIC_DRIVERS_VERSION)"
```

## Starkes Rauschen

1. **Flackerndes Licht prüfen:** LED- und Leuchtstofflampen flackern mit 100 Hz und erzeugen großflächig Events. Teste bei Tageslicht oder mit einer Glühlampe.
2. **Mehr Licht:** In dunklen Szenen rauschen Eventkameras stark.
3. **Biases erhöhen:** `"biases": {"diff_on": 30, "diff_off": 30}`, schrittweise anpassen (siehe [Konfiguration](konfiguration.md#biases-se1-s4)).
4. **Rauschfilter einschalten:** `"filter": {"background_activity": {"enabled": true}}`. Stärker filtern mit kleinerem `time_us` oder `min_neighbors: 2`.

## Hohe Event-Raten, Verzögerung, verlorene Daten

Wächst `packet.status.ring.backlog` ständig, ist die Verarbeitung langsamer als die Kamera.

- Weniger Events erzeugen: Biases `diff_on`/`diff_off` erhöhen, Bildbereich mit Masken einschränken, Flackern vermeiden.
- Rate-Limiter im Sensor: `"rate_limiter": {"reference_period_us": 200, "maximum_events_per_period": 4000}`
- Verarbeitung beschleunigen: mit NumPy-Arrays (`packet.events`) arbeiten statt mit `rows()` oder `to_csv()` pro Event.
- Weniger anzeigen: Frames seltener darstellen. Die Anzeige mit OpenCV kostet oft mehr Zeit als das Lesen.

## Mehrere Kameras: Zeitstempel laufen auseinander

Ohne Synchronisation hat jede Kamera ihre eigene Uhr. Lösung: Sync-Kabel und `clock`-Einstellung, siehe [Mehrere Kameras](mehrere-kameras.md#uhren-synchronisieren).

## Konfiguration wird abgelehnt

`opennve check datei.json` zeigt die Ursache mit dem betroffenen Feld. Häufige Fälle:

| Meldung | Ursache |
|---|---|
| `serial must be a string` | Seriennummer in Anführungszeichen schreiben: `"05110634"` |
| `unknown setting(s) in settings` | Tippfehler im Feldnamen. Gültige Namen zeigt `opennve template <modell>` |
| `output field c is not supported` | Der Kameraname steht einmal pro Paket in `packet.camera`, nicht in den Events |
| `duplicate camera name` | Jeder `name` darf in einer Datei nur einmal vorkommen |
