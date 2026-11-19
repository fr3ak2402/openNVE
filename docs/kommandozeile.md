# Kommandozeile

Nach der Installation gibt es den Befehl `opennve` (alternativ `python -m opennve`).

## `opennve list`

Zeigt die angeschlossenen Kameras.

```sh
opennve list
opennve list --config cameras.json   # mit den Namen aus der Konfiguration
opennve list --json                  # als JSON
```

```text
SE1-S4     05110634   USB 3.0 SuperSpeed (5.0 Gb/s)    ok (left)
SE1-S4     05110636   USB 3.0 SuperSpeed (5.0 Gb/s)    ok (right)
```

Steht statt `ok` ein Fehler, kann die Kamera nicht benutzt werden, siehe [Fehlerbehebung](fehlerbehebung.md).

## `opennve template <modell>`

Gibt eine vollständige Konfiguration mit allen Standardwerten aus.

```sh
opennve template SE1-S4
opennve template SE1-S4 --serial 05110634 --name left > left.json
```

## `opennve check <datei>`

Prüft eine Konfigurationsdatei, ohne eine Kamera zu öffnen.

```sh
opennve check cameras.json
```

```text
left: SE1-S4 05110634
right: SE1-S4 05110636
2 camera(s), configuration is valid
```

Bei Fehlern nennt die Meldung die Kamera und das Feld, z. B. `cameras.json: camera 1: settings.clock must be one of internal, internal_with_output_enabled, external`. Der Rückgabewert ist dann `1`.

## `opennve models`

Listet die unterstützten Modelle.

## Befehle des eingebetteten Treibers

| Befehl | Bedeutung |
|---|---|
| `ndls` | Angeschlossene Geräte (Tabelle von `neuromorphic_drivers`) |
| `neuromorphic-drivers-install-windows-driver` | WinUSB-Treiber unter Windows installieren (`--list`, `--device VID:PID`, `--force`) |
| `neuromorphic-drivers-install-udev-rules` | udev-Regeln unter Linux installieren |
