# Addition vs. Multiplikation — eigene Auswertung der `txf_mul_*`-Läufe

```
STATUS: VORLAGE — noch kein Text des Autors. Vor dem Push entweder ausfüllen oder löschen.
```

Diese Datei ist der Ort für die Auswertung, die `docs/HUMAN_DECISIONS.md` **E6** dem menschlichen Autor
vorbehält. Die KI hat die drei Läufe nicht analysiert und analysiert sie nicht. Nur die Struktur unten stammt
von der KI (2026-09-20); jeder Absatz mit `[HENRIK]` wird vom Autor selbst geschrieben. Zahlen werden aus
der Werkzeug-Ausgabe kopiert, nicht aus dem Gedächtnis.

## 0. Was ausgewertet wurde

| | |
|---|---|
| Läufe | `txf_mul_p113_wd1.0_frac0.5_gf2.0_seed{0,1,2}` (Multiplikation) gegen `txf_add_p113_wd1.0_frac0.5_gf2.0_seed{0,1,2}` (Addition, **gleiche Einstellung, gleiche Seeds → identische Initialisierung**) |
| Werkzeug | `bwki/experiment_add_vs_mult.py --analyze-run <run>` — liest `embeddings.npy` (erster und letzter Checkpoint), rechnet Top-8-Anteil, normierte Spektralentropie und Participation Ratio, jeweils in **natürlicher Ordnung** (0…112) und in **Primitivwurzel-Ordnung** (g = 3: 3⁰, 3¹, 3², … mod 113) |
| Null-Baseline | flaches Spektrum: Top-8 = 8/56 = 0,143, Entropie ≈ 1,0, Participation Ratio ≈ 56 |
| Befehl | `[HENRIK: den ausgeführten Befehl hier einfügen]` |
| Datum der Auswertung | `[HENRIK]` |

## 1. Fragen — vor dem Blick auf die Zahlen hingeschrieben

- **F1** Generalisiert das Netz bei Multiplikation überhaupt (Test-Accuracy am Ende, aus `run.json`)?
- **F2** Ist das mul-Embedding in *natürlicher* Ordnung spektral flach (nahe Baseline)?
- **F3** Wird es in *Primitivwurzel*-Ordnung spärlich — in derselben Größenordnung wie das add-Embedding in natürlicher Ordnung?
- **F4** Generalisiert mul später als add bei gleichem Seed, und ist der Abstand größer als die Streuung zwischen den Seeds?

Hintergrund (Power et al. 2022 §3.2; `docs/sources/power2022_and_grokfast2024.md` Punkt 8): Multiplikation der
Reste ≠ 0 ist über den diskreten Logarithmus zur Basis g isomorph zur Addition mod 112. Wenn F2 und F3 beide
zutreffen, ist das konsistent damit, dass das Netz die Aufgabe *in der Log-Basis* löst.

## 2. Zahlen

`[HENRIK: Tabelle aus der Werkzeug-Ausgabe hier einfügen — alle sechs Läufe, init und final, beide Ordnungen]`

| run | checkpoint | ordering | top8 | entropy_norm | particip_ratio | test acc (run.json) | generalisiert bei Schritt |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## 3. Was ich sehe — Beobachtung ohne Deutung

`[HENRIK]`

## 4. Was ich daraus schließe — und was nicht

`[HENRIK]`

Mindestens diese Alternativen benennen: nur 3 Seeds; Grokfast an und `train_frac` 0,5 (nicht die
un-beschleunigte Referenz); Token 0 ist bei mul ausgeschlossen, bei add nicht (die natürlichen Top-8-Werte sind
deshalb nicht 1:1 vergleichbar); Top-8 ist eine feste Kappe (`docs/LEGACY_METRIC_AUDIT.md`), daher Entropie und
Participation Ratio daneben lesen; die Transition liegt auf einem Log-Gitter (Intervall, kein Punkt).

## 5. Wer was gemacht hat

| | |
|---|---|
| KI | Skript `experiment_add_vs_mult.py` (Entwurf von Ali Kandora, einem Kollegen des Autors; Fourier-Basis-Fehler bei geradem n von der KI korrigiert und per Self-Test verifiziert), der Modus `--analyze-run`, diese Vorlage |
| Autor | `[HENRIK: was du selbst gemacht hast — Frage gestellt, Befehl ausgeführt, Zahlen gelesen, Abschnitte 3 und 4 geschrieben, …]` |

## Signatur

| | |
|---|---|
| Name | `[HENRIK]` |
| Datum | `[HENRIK]` |
| Abschnitte 3 und 4 sind in meinen eigenen Worten geschrieben | ja / nein |
