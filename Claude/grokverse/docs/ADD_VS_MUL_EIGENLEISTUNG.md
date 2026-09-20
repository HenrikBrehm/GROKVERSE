# Addition vs. Multiplikation — eigene Auswertung der `txf_mul_*`-Läufe

```
STATUS: ZAHLEN GEMESSEN (KI, 2026-09-20) — Abschnitte 3 und 4 und die Signatur fehlen noch.
```

> **Änderung der Zuständigkeit am 2026-09-20.** `docs/HUMAN_DECISIONS.md` **E6** hatte diese drei Läufe
> der Auswertung durch den Autor vorbehalten. Der Autor hat diese Reservierung am 2026-09-20
> ausdrücklich aufgehoben („ignore the e6 rule"). Die **Messung** in Abschnitt 2 wurde daraufhin von der
> KI ausgeführt und ist als KI-Arbeit zu führen — `AI_DISCLOSURE.md` §5 ist entsprechend korrigiert.
> **Nicht** von der KI geschrieben werden die Abschnitte 3 und 4 und die Signatur: das ist die
> Deutung in eigenen Worten, und sie als Eigenleistung auszuweisen wäre nur dann richtig, wenn sie
> auch vom Autor stammt. Sie bleiben offen.

Zahlen sind aus der Werkzeug-Ausgabe kopiert, nicht aus dem Gedächtnis.

## 0. Was ausgewertet wurde

| | |
|---|---|
| Läufe | `txf_mul_p113_wd1.0_frac0.5_gf2.0_seed{0,1,2}` (Multiplikation) gegen `txf_add_p113_wd1.0_frac0.5_gf2.0_seed{0,1,2}` (Addition, **gleiche Einstellung, gleiche Seeds → identische Initialisierung**) |
| Werkzeug | `bwki/experiment_add_vs_mult.py --analyze-run <run>` — liest `embeddings.npy` (erster und letzter Checkpoint), rechnet Top-8-Anteil, normierte Spektralentropie und Participation Ratio, jeweils in **natürlicher Ordnung** (0…112) und in **Primitivwurzel-Ordnung** (g = 3: 3⁰, 3¹, 3², … mod 113) |
| Null-Baseline | flaches Spektrum: Top-8 = 8/56 = 0,143, Entropie ≈ 1,0, Participation Ratio ≈ 56 |
| Befehl | `python experiment_add_vs_mult.py --analyze-run training/runs/txf_mul_p113_wd1.0_frac0.5_gf2.0_seed{0,1,2} --analyze-run training/runs/txf_add_p113_wd1.0_frac0.5_gf2.0_seed{0,1,2}` (sechs `--analyze-run`-Angaben, aus `bwki/`) |
| Datum der Auswertung | 2026-09-20, KI (siehe Kasten oben) |
| Rohausgabe | `analyze_run_results.json` (vom Werkzeug geschrieben) |

## 1. Fragen — vor dem Blick auf die Zahlen hingeschrieben

- **F1** Generalisiert das Netz bei Multiplikation überhaupt (Test-Accuracy am Ende, aus `run.json`)?
- **F2** Ist das mul-Embedding in *natürlicher* Ordnung spektral flach (nahe Baseline)?
- **F3** Wird es in *Primitivwurzel*-Ordnung spärlich — in derselben Größenordnung wie das add-Embedding in natürlicher Ordnung?
- **F4** Generalisiert mul später als add bei gleichem Seed, und ist der Abstand größer als die Streuung zwischen den Seeds?

Hintergrund (Power et al. 2022 §3.2; `docs/sources/power2022_and_grokfast2024.md` Punkt 8): Multiplikation der
Reste ≠ 0 ist über den diskreten Logarithmus zur Basis g isomorph zur Addition mod 112. Wenn F2 und F3 beide
zutreffen, ist das konsistent damit, dass das Netz die Aufgabe *in der Log-Basis* löst.

## 2. Zahlen

Null-Baseline zum Vergleich: Top-8 = 0,143 · Entropie = 1,000 · PR = 56,0.

**Multiplikation** (`txf_mul_p113_wd1.0_frac0.5_gf2.0_seed*`):

| seed | checkpoint | ordering | top8 | entropy_norm | PR | test acc | generalisiert bei Schritt |
|---|---|---|---|---|---|---|---|
| 0 | init | natural | 0,1625 | 0,9984 | 55,33 | | |
| 0 | init | log_permuted | 0,1644 | 0,9981 | 55,24 | | |
| 0 | final | natural | **0,1779** | 0,9972 | 54,80 | 0,9850 | 912 |
| 0 | final | log_permuted | **0,5704** | 0,8414 | 16,56 | | |
| 1 | init | natural | 0,1606 | 0,9987 | 55,48 | | |
| 1 | init | log_permuted | 0,1646 | 0,9982 | 55,28 | | |
| 1 | final | natural | **0,1780** | 0,9969 | 54,74 | 0,9887 | 912 |
| 1 | final | log_permuted | **0,5382** | 0,8722 | 20,39 | | |
| 2 | init | natural | 0,1664 | 0,9979 | 55,13 | | |
| 2 | init | log_permuted | 0,1646 | 0,9985 | 55,36 | | |
| 2 | final | natural | **0,1706** | 0,9971 | 54,84 | 0,9826 | 1029 |
| 2 | final | log_permuted | **0,5521** | 0,8528 | 18,38 | | |

**Addition**, gleiche Einstellung und gleiche Seeds (`txf_add_p113_wd1.0_frac0.5_gf2.0_seed*`):

| seed | checkpoint | ordering | top8 | entropy_norm | PR | test acc | generalisiert bei Schritt |
|---|---|---|---|---|---|---|---|
| 0 | init | natural | 0,1609 | 0,9992 | 55,63 | | |
| 0 | init | log_permuted | 0,1644 | 0,9981 | 55,24 | | |
| 0 | final | natural | **0,5206** | 0,8687 | 18,54 | 0,9834 | 675 |
| 0 | final | log_permuted | **0,1687** | 0,9983 | 55,27 | | |
| 1 | init | natural | 0,1610 | 0,9992 | 55,65 | | |
| 1 | init | log_permuted | 0,1646 | 0,9982 | 55,28 | | |
| 1 | final | natural | **0,6050** | 0,7914 | 10,32 | 0,9904 | 859 |
| 1 | final | log_permuted | **0,1647** | 0,9986 | 55,39 | | |
| 2 | init | natural | 0,1640 | 0,9988 | 55,47 | | |
| 2 | init | log_permuted | 0,1646 | 0,9985 | 55,36 | | |
| 2 | final | natural | **0,6432** | 0,7349 | 7,48 | 0,9850 | 716 |
| 2 | final | log_permuted | **0,1634** | 0,9986 | 55,39 | | |

**Zeitpunkt der Generalisierung, paarweise bei gleichem Seed** (gleiche Initialisierung; Log-Gitter,
deshalb Intervalle, keine Punkte):

| seed | add | mul | mul − add |
|---|---|---|---|
| 0 | 675 | 912 | +237 |
| 1 | 859 | 912 | +53 |
| 2 | 716 | 1029 | +313 |

Streuung zwischen den Seeds: add 675–859 (Spanne 184), mul 912–1029 (Spanne 117).

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
