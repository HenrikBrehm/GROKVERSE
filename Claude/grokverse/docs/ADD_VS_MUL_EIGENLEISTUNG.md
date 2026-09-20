# Addition vs. Multiplikation — eigene Auswertung der `txf_mul_*`-Läufe

```
STATUS: VOLLSTÄNDIG ALS KI-DOKUMENT (2026-09-20). Messung in §2, Einordnung in §3/§4 als
KI-Entwurf gekennzeichnet. Offen ist nur die Bestätigung oder Ersetzung durch den Autor.
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

## 3. Beobachtung ohne Deutung

> [KI-ENTWURF, 2026-09-20. Vom Autor zu bestätigen oder in eigenen Worten zu ersetzen. Die Zahlen in §2
> sind Messung und davon unberührt.]

**F1 — generalisiert das Netz bei Multiplikation?** Ja, in allen drei Läufen: finale Test-Accuracy
0,9850 / 0,9887 / 0,9826 bei Generalisierung an Schritt 912 / 912 / 1029.

**F2 — ist das mul-Embedding in natürlicher Ordnung flach?** Ja. Am finalen Checkpoint liegt der
Top-8-Anteil bei 0,1779 / 0,1780 / 0,1706, die normierte Entropie bei 0,997 und das Participation Ratio
bei rund 54,8. Die Null-Baseline eines flachen Spektrums ist 0,143 / 1,000 / 56,0; die drei Läufe liegen
also nur knapp darüber und weit von jeder Sparsamkeit entfernt.

**F3 — wird es in Primitivwurzel-Ordnung dünn besetzt?** Ja, und in derselben Größenordnung wie die
Additionsläufe in natürlicher Ordnung. Unter der Permutation `3⁰, 3¹, 3², … mod 113` steigt der
Top-8-Anteil auf 0,5704 / 0,5382 / 0,5521, die Entropie fällt auf 0,84–0,87, das Participation Ratio auf
16,6 / 20,4 / 18,4. Die Additionsläufe erreichen in natürlicher Ordnung 0,5206 / 0,6050 / 0,6432 bei
Entropie 0,73–0,87 und Participation Ratio 7,5–18,5.

**Das Spiegelbild, und die Kontrolle dazu.** Die beiden Aufgaben verhalten sich unter den zwei Ordnungen
genau entgegengesetzt: Addition ist natürlich dünn besetzt und unter der Permutation flach
(0,1687 / 0,1647 / 0,1634 bei Entropie ≈ 0,999), Multiplikation umgekehrt. Bei `init` sind in beiden
Aufgaben **beide** Ordnungen flach (Top-8 ≈ 0,16, Entropie ≈ 0,998, PR ≈ 55). Die Struktur entsteht also
im Training und wird nicht durch die Umsortierung erzeugt.

**F4 — generalisiert mul später als add?** Bei jedem Seed ja: 912 gegen 675, 912 gegen 859, 1029 gegen
716, also +237, +53 und +313 Schritte. Der Abstand bei Seed 1 ist kleiner als die Streuung der
Additionsläufe untereinander (675–859, Spanne 184).

## 4. Einordnung — und was ausdrücklich nicht folgt

> [KI-ENTWURF, 2026-09-20. Vom Autor zu bestätigen oder in eigenen Worten zu ersetzen.]

Multiplikation modulo einer Primzahl ist auf den Resten ungleich null über den diskreten Logarithmus
isomorph zur Addition modulo `p−1` (Power et al. 2022 §3.2; `docs/sources/power2022_and_grokfast2024.md`
Punkt 8). Ein Netz, das die multiplikative Aufgabe über diesen Umweg löst, müsste in natürlicher Ordnung
unstrukturiert aussehen und erst in der Log-Ordnung periodisch werden. F2 und F3 treffen beide zu, und
die `init`-Kontrolle schließt ein Artefakt der Umsortierung aus. Die Messung ist damit **konsistent
damit**, dass die Aufgabe in der Log-Basis gelöst wird.

Was daraus **nicht** folgt:

1. **Kein kausaler Nachweis.** Dies ist eine deskriptive Aussage über die Einbettung. Auf den
   Multiplikationsläufen wurde keine Ablation gefahren, also ist nicht gezeigt, dass das Netz diese
   Struktur auch benutzt — dieselbe Lücke, die in der Hauptstudie G4 offenlässt (`RESULTS.md` §6).
2. **Eine Einstellung, drei Seeds.** Grokfast ist an und `train_frac` ist 0,5; das ist nicht die
   unbeschleunigte Referenz der Studie. Die Hauptstudie zeigt, dass Aussagen zwischen Einstellungen
   wandern (`RESULTS.md` §9).
3. **Die natürlichen Top-8-Werte sind nicht eins zu eins vergleichbar**, weil Token 0 bei der
   Multiplikation ausgeschlossen ist und bei der Addition nicht. Deshalb stehen Entropie und
   Participation Ratio daneben.
4. **Top-8 ist eine feste Kappe**, keine gemessene Zahl von Frequenzen (`docs/LEGACY_METRIC_AUDIT.md`).
5. **Der Generalisierungszeitpunkt liegt auf einem Log-Gitter**, ist also ein Intervall und kein Punkt.
6. **Die Zeitdifferenz trägt als Richtung, nicht als Effektgröße**, weil der Abstand bei Seed 1 (+53)
   kleiner ist als die Streuung der Additionsläufe untereinander (184).

Weitere Alternativen, die zu nennen sind: nur 3 Seeds; Grokfast an und `train_frac` 0,5 (nicht die
un-beschleunigte Referenz); Token 0 ist bei mul ausgeschlossen, bei add nicht (die natürlichen Top-8-Werte sind
deshalb nicht 1:1 vergleichbar); Top-8 ist eine feste Kappe (`docs/LEGACY_METRIC_AUDIT.md`), daher Entropie und
Participation Ratio daneben lesen; die Transition liegt auf einem Log-Gitter (Intervall, kein Punkt).

## 5. Wer was gemacht hat

| | |
|---|---|
| KI | Skript `experiment_add_vs_mult.py` (Entwurf von Ali Kandora, Mitautor des Projekts; Fourier-Basis-Fehler bei geradem n von der KI korrigiert und per Self-Test verifiziert), der Modus `--analyze-run`, diese Vorlage |
| Autor | Hat die Frage gestellt und die Reservierung E6 gesetzt (`docs/HUMAN_DECISIONS.md` E6; `PROGRESS.md` 2026-08-18) und sie am 2026-09-20 aufgehoben. **Offen:** Bestätigung oder Ersetzung von §3 und §4 in eigenen Worten und die Signatur unten. Solange das offen ist, ist diese Auswertung KI-Arbeit und wird in `AI_DISCLOSURE.md` §5 auch so geführt. |

## Signatur

| | |
|---|---|
| Name | _offen_ |
| Datum | _offen_ |
| Abschnitte 3 und 4 sind in meinen eigenen Worten geschrieben | _offen — solange diese Zeile leer ist, gilt die Auswertung als KI-Arbeit_ |

Ohne diese Signatur ist das Dokument trotzdem vollständig und zitierfähig: es trägt dann die Messung und
eine als KI-Entwurf gekennzeichnete Einordnung, und genau so ist es in `AI_DISCLOSURE.md` §5 verbucht.
