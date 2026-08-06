# GROKVERSE — Study Guide (interne Vorbereitung, NICHT einreichen)

> Ziel dieses Dokuments: Du (Henrik) sollst **jedes** Konzept, jede Zahl und jede
> Designentscheidung in GROKVERSE so verstehen, dass du es vor einem ML-Forscher
> (Tübingen/MPI) am Whiteboard herleiten und verteidigen kannst. Das ist der
> entscheidende Hebel für die BWKI-Bewertung (Eigenständigkeit steht an erster
> Stelle). Lies das langsam, rechne die Herleitungen selbst nach, und erkläre sie
> einer Person, die kein ML kann. Wenn du das kannst, ist der „KI-gebaut"-Einwand
> der Jury weitgehend entschärft.

**Dieses Dokument gehört NICHT in das eingereichte Repo.** Es liegt absichtlich in
`bwki/PREP/`, außerhalb von `grokverse/`.

---

## 0. Das ganze Projekt in drei Sätzen

1. Wir reproduzieren **Grokking**: ein winziges neuronales Netz merkt sich seine
   Trainingsdaten erst auswendig (Train-Genauigkeit 100 %), versagt lange auf
   neuen Daten — und generalisiert dann **plötzlich**, tausende Schritte später.
2. Wir zeigen mit **mechanistischer Interpretierbarkeit**, *wie* es das tut: das
   Netz entdeckt, dass modulare Addition mit Sinus/Cosinus-Wellen (einer
   „Trig-Identität") gerechnet werden kann, und wir messen genau diese Struktur.
3. Wir machen das Ganze in einem **interaktiven 3D-Explorer** sicht- und
   anfassbar — plus ein eigenes Experiment (Transformer vs. MLP) als
   Erkenntnisgewinn.

**Der Ein-Satz-Hook:** „Schau dem Moment zu, in dem eine KI es *kapiert*."

---

## 1. Grokking — was es ist und warum es erstaunlich ist

**Normalerweise** wird ein Netz beim Training gleichmäßig besser: Trainings- und
Testleistung steigen zusammen. **Grokking** (Power et al. 2022) bricht das:

- **Phase 1 — Memorisierung:** Das Netz lernt die Trainingsbeispiele *auswendig*.
  Train-Accuracy → 100 %, Test-Accuracy bleibt bei Zufallsniveau. Es hat eine
  Nachschlagetabelle gebaut, keine Regel.
- **Phase 2 — Plateau:** Lange (bei uns ~8000 Schritte) scheint nichts zu
  passieren. Train-Accuracy bleibt 100 %, Test-Accuracy bleibt zufällig. Aber
  *innen* formt sich etwas (das messen wir später!).
- **Phase 3 — Generalisierung („der Grok"):** Plötzlich springt die
  Test-Accuracy von ~Zufall auf ~100 %. Das Netz hat die *zugrundeliegende Regel*
  gefunden und die Auswendiglösung durch einen echten Algorithmus ersetzt.

**Warum ist das wichtig (nicht nur ein Kuriosum)?** Grokking ist eines der
saubersten Fenster in die Frage „lernt ein Netz eine echte Regel oder merkt es
sich nur Daten?". Genau diese Frage ist zentral für **KI-Sicherheit und
Interpretierbarkeit** großer Modelle. Hier ist das Netz klein genug, um die Regel
*vollständig auszulesen* — das geht bei GPT-Klassen-Modellen (noch) nicht.

**Unsere gemessenen Zahlen (un-beschleunigter kanonischer Lauf), die du auswendig können solltest:**
- Train-Accuracy → 1.000 bei **Schritt 145** (Memorisierung).
- Test-Accuracy kreuzt 0.95 bei **Schritt 8367** (Generalisierung).
- **Grok-Gap = 8222 Schritte.** Finale Test-Accuracy **0.981**.
- Test-Loss: 4.7 → 0.06, steigt im Plateau zwischenzeitlich auf ~26 (das Netz
  wird *überzeugter falsch*, bevor es kippt).

> **30-Sekunden-Erklärung:** „Stell dir einen Schüler vor, der erst alle
> Lösungen einer Übungsklausur auswendig lernt — in der echten Klausur fällt er
> durch. Dann, viel später, *versteht* er plötzlich die Regel dahinter und kann
> jede neue Aufgabe lösen. Genau das macht unser Netz, und wir filmen den Moment
> des Verstehens in Zeitlupe."

---

## 2. Die Aufgabe: modulare Addition `(a + b) mod p`, p = 113

**Was ist modulare Addition?** Rechnen „mit Uhr-Überlauf": `(a + b) mod p` ist der
Rest von `a+b` bei Division durch `p`. Beispiel mod 12 (Uhr): `10 + 5 = 15 ≡ 3`.
Wir nehmen `p = 113` (eine Primzahl).

**Warum diese Aufgabe?** (Wichtige Verteidigungsfrage!)
- **Klein genug, um sie komplett zu verstehen:** Es gibt nur `113 × 113 = 12.769`
  mögliche Aufgaben. Wir kennen die wahre Regel exakt.
- **Reich genug, um zu grokken:** Sie hat eine echte algebraische Struktur
  (zyklische Gruppe ℤ₁₁₃), die das Netz „entdecken" kann.
- **Periodisch:** Modulare Addition ist von Natur aus zyklisch/periodisch — das
  macht die Fourier-Analyse natürlich (siehe §5). `p` prim sorgt dafür, dass alle
  Frequenzen `k=1..(p-1)/2` „sauber" sind.
- **Es ist das Standard-Rezept von Nanda et al. 2023** — wir reproduzieren
  bewusst die kanonische Aufgabe, statt eine neue zu erfinden, damit unsere
  Reproduktion direkt mit der Literatur vergleichbar ist.

**Datenformat:** Jede Aufgabe ist die Token-Sequenz `[a, b, =]` (drei Tokens), und
das Label ist `c = (a+b) mod p`. Das Vokabular hat `p+1 = 114` Tokens: die Zahlen
`0..112` plus ein spezielles `=`-Token (Index 113). Die Vorhersage wird an der
Position des `=`-Tokens abgelesen.

**Train/Test-Split:** Wir nehmen einen festen Anteil `train_frac` (kanonisch 0.3,
also 30 %) der 12.769 Aufgaben zum Trainieren, der Rest ist Test. Der Split ist
**seed-deterministisch** (eine feste Permutation, abhängig vom Seed) — gleiche
Seed ⇒ gleicher Split ⇒ reproduzierbar. *Warum nur 30 %?* Mit zu vielen
Trainingsdaten ist Auswendiglernen zu einfach und es entsteht weniger Druck zu
generalisieren; `train_frac` ist einer der Knöpfe, die Grokking steuern.

---

## 3. Das Modell: 1-Layer-Transformer (Nanda-Rezept)

Bewusst **minimal**, damit die gelernte Schaltung lesbar ist. Datei:
`training/grokverse/models/transformer.py`. Dimensionen: `d_model=128`,
`n_heads=4`, `d_head=32` (4×32=128), `d_mlp=512`, **kein LayerNorm**.

**Der Vorwärtspfad, Schritt für Schritt** (du solltest jeden Block erklären können):

1. **Embedding:** Jedes Token wird über `W_E` (Matrix `[114, 128]`) in einen
   128-dim Vektor übersetzt; dazu kommt ein **Positions-Embedding** `W_pos`
   (Position 0/1/2). → `x = W_E[token] + W_pos`. Die Matrix `W_E` ist *das
   Hauptobjekt unserer Analyse* (§5).
2. **Attention (4 Köpfe):** Für jeden Kopf werden aus `x` per `W_Q, W_K, W_V`
   Query/Key/Value berechnet. Die Attention-Gewichte sind
   `softmax(Q·Kᵀ / √d_head)` mit **kausaler Maske** (Position darf nur auf sich
   und frühere schauen). Die `=`-Position (die letzte) kann also auf `a`, `b` und
   sich selbst schauen. Output: `x = x + Σ_heads (attn · V) · W_O`.
3. **MLP:** `x = x + ReLU(x·W_in)·W_out` (eine versteckte Schicht, Breite 512).
   *Hier* entstehen die nichtlinearen **Produkte** von Frequenzen, die die
   Trig-Identität braucht (§5.2).
4. **Unembedding:** `logits = x·W_U` (`[128, 114]`). Wir lesen nur die
   Logits an der `=`-Position und nur die ersten `p=113` Klassen ab (das
   `=`-Token kann nie die Antwort sein).

**Warum kein LayerNorm?** LayerNorm würde die Skalen verwischen und die
Fourier-Struktur schwerer auslesbar machen. Nanda lässt es weg, damit die
Schaltung *mathematisch sauber* interpretierbar ist. Das ist ein bewusster
Interpretierbarkeits-Trade-off, kein Versehen.

**Gewichtsinitialisierung:** normalverteilt mit Std `init_scale/√(fan_in)` — eine
Standard-Skalierung, damit die Aktivierungen anfangs eine vernünftige Größe haben.

---

## 4. Das Training — und warum Weight Decay Grokking *verursacht*

Datei: `training/grokverse/train.py`. Setup:
- **Full-batch AdamW** (der gesamte Trainingssatz ist ein Batch — kein
  Mini-Batch-Rauschen, sauberere Dynamik), `lr=1e-3`, `betas=(0.9, 0.98)`.
- **`weight_decay = 1.0`** — ungewöhnlich hoch. Das ist der Schlüssel.
- **Logarithmische Logging-Schedule:** Wir loggen bei Schritten 0,1,2,5,10,20,…
  (geomspace). Grokking spielt sich über mehrere Größenordnungen ab; linear
  loggen würde den frühen Teil verschwenden und den Übergang verpassen.

**Warum grokt es? (Die zentrale „Warum"-Frage — Omnigrok, Liu et al. 2022)**

Es gibt zwei Lösungen für die Aufgabe:
- die **Auswendig-Lösung** (Nachschlagetabelle): schnell zu finden, aber sie
  braucht *große* Gewichte (viele unabhängige Einträge) und generalisiert nicht.
- die **Generalisierungs-Lösung** (die Trig-Schaltung): schwerer zu finden, aber
  sie hat eine *kleine Gewichtsnorm* und löst alle Aufgaben.

**Weight Decay** drückt permanent die Gewichtsnorm Richtung null. Am Anfang ist
Auswendiglernen der schnellste Weg, den Trainings-Loss zu senken — also macht das
Netz das zuerst (Phase 1). Aber Weight Decay „bestraft" die großen Gewichte der
Auswendig-Lösung. Über das lange Plateau schiebt es das Netz langsam in Richtung
der *normärmeren* Trig-Lösung. Wenn die Trig-Schaltung gut genug ausgebildet ist,
übernimmt sie — und die Test-Accuracy springt (Phase 3). **Grokking ist der
Wettlauf zwischen schnellem Auswendiglernen und langsam wachsendem, durch Weight
Decay getriebenem Generalisieren.** (Das ist die Omnigrok-Sicht: Grokking wird von
der Gewichtsnorm gesteuert.)

> **Killer-Antwort, wenn die Jury fragt „warum grokt es?":** „Weil zwei Lösungen
> konkurrieren — eine auswendige mit großer Gewichtsnorm und eine algorithmische
> mit kleiner. Weight Decay bestraft die Norm, also gewinnt über die Zeit die
> algorithmische. Liu et al. (Omnigrok) zeigen genau das: Grokking ist durch die
> Gewichtsnorm kontrolliert."

---

## 5. Mechanistische Analyse — das Herzstück

### 5.1 Fourier auf ℤ_p — was die Embeddings „sind"

Eine **Fourier-Basis** über den Zahlen `0..p-1` besteht aus dem konstanten Modus
plus, für jede Frequenz `k = 1..(p-1)/2`, einem Cosinus `cos(2πkn/p)` und einem
Sinus `sin(2πkn/p)` (Datei `analysis/fourier.py`, Funktion `fourier_basis`). Diese
Basis ist **orthonormal** (das testen wir explizit, `test_core.py`).

Wir nehmen die finale Embedding-Matrix `W_E[:113]` (eine 128-dim Repräsentation
jeder Zahl 0..112) und **projizieren sie auf diese Basis**. Ergebnis: fast die
*gesamte* „Power" (Energie) sitzt auf **wenigen** Frequenzen.

- **Gemessen:** Die **top-8 Frequenzen halten ~76 %** der Frequenz-Power (beim
  kanonischen Lauf), gegenüber nur ~32 % bei einem nicht-grokkten Lauf.
- Das bedeutet: Das Netz repräsentiert jede Zahl `n` ungefähr als
  `(cos(w_k n), sin(w_k n))` für eine Handvoll Schlüsselfrequenzen `w_k = 2πk/p`.
  Geometrisch: jede Zahl liegt auf einem **Kreis**, im Winkel `w_k n`.

> **Was „76 % top-8 power" konkret heißt:** Von der gesamten oszillierenden
> Energie in den Embeddings stecken drei Viertel in nur acht Wellenlängen. Das ist
> keine Deko — das ist der Fingerabdruck eines *sparsamen, periodischen
> Algorithmus*. Ein Auswendiglerner hätte breit verteilte, strukturlose Power.

### 5.2 Die Trig-Identitäts-Schaltung — DIE Herleitung (musst du können!)

Warum hilft Periodizität bei `(a+b) mod p`? Wegen der **Additionstheoreme**:

```
cos(w(a+b)) = cos(wa)·cos(wb) − sin(wa)·sin(wb)
sin(w(a+b)) = sin(wa)·cos(wb) + cos(wa)·sin(wb)
```

Das Netz hat `cos(wa), sin(wa)` (aus dem Embedding von `a`) und `cos(wb), sin(wb)`
(aus `b`). Der **MLP-Block bildet die Produkte** (z. B. `cos(wa)·cos(wb)`) — eine
Schicht mit Nichtlinearität kann solche multiplikativen Wechselwirkungen
darstellen. Damit berechnet das Netz `cos(w(a+b))` und `sin(w(a+b))`, ohne `a+b`
je „direkt" addiert zu haben.

Zum Schluss liest das **Unembedding** für jede Kandidatenantwort `c` aus:

```
logit(c) ∝ Σ_k [ cos(w_k(a+b))·cos(w_k c) + sin(w_k(a+b))·sin(w_k c) ]
         = Σ_k cos( w_k (a + b − c) )
```

(im letzten Schritt wieder ein Additionstheorem). Diese Summe ist **maximal, wenn
`c ≡ a+b (mod p)`** — dann ist jeder Term `cos(0) = 1`, alle Wellen
interferieren konstruktiv. Für jedes andere `c` heben sich die Wellen weitgehend
auf (destruktive Interferenz). Also wählt `argmax_c logit(c)` genau `(a+b) mod p`.
**Das ist der Algorithmus, den das Netz „erfindet".**

> Wenn du **eine** Sache am Whiteboard können musst, dann diese Herleitung:
> Embedding als Winkel → MLP bildet Produkte → Additionstheorem →
> konstruktive Interferenz bei `c = a+b`. Übe das laut.

### 5.3 PCA-Ring

PCA (Hauptkomponentenanalyse) projiziert die 128-dim Embeddings auf 3 Dimensionen
(`analysis/pca.py`). Weil die Embeddings periodisch sind, liegen die Zahlen nach
dem Grokken sichtbar auf einem **Ring** (die geometrische Seite von §5.1). Wichtig:
Wir fitten die PCA **einmal** auf die finalen Embeddings und wenden dieselbe
Projektion auf *alle* Zeitschritte an — deshalb sieht man im Explorer, wie sich
der „Blob" über das Training in den Ring zusammenzieht. (Sonst würde jede
Zeitscheibe in einem anderen Koordinatensystem leben und die Animation wäre
bedeutungslos.)

### 5.4 Attention — die Schaltung mit der Rechnung verknüpfen

Gemittelt über alle 12.769 Eingaben achtet die `=`-Position (wo abgelesen wird)
**~50/50 auf beide Operanden** `a` und `b` und praktisch gar nicht auf sich
selbst (pro Kopf ≈ 0.50 / 0.50 / 0.001). Das ist genau, was man erwartet: um
`(a+b)` zu berechnen, *muss* das Netz beide Operanden gleich gewichtet
heranziehen. Das verbindet die gelernte Struktur direkt mit der Berechnung.

### 5.5 Restricted & Excluded Loss — die Nanda-Progress-Measures (NEU)

Datei: `analysis/progress_measures.py`. **Das ist die namensgebende Methode des
Papers, das wir reproduzieren** (Nanda et al. 2023: „Progress Measures…"). Idee:
ein **held-out-unabhängiges** Maß für „wie weit ist die Schaltung?", das *nicht*
auf die Test-Accuracy schaut.

Wir nehmen die Logits des Netzes über das ganze `(a,b)`-Gitter, `L[a,b,c]`, und
**Fourier-transformieren sie 2D über die Eingabe-Achsen** `a` und `b`. Dann:

- **Restricted Loss:** Behalte *nur* die Komponenten auf den Schlüsselfrequenzen
  (plus Konstante), baue die Logits daraus neu, miss den Loss. *Wenn* die echte
  Rechnung die Trig-Schaltung ist, lösen schon diese wenigen Frequenzen die
  Aufgabe → Restricted Loss fällt während der Schaltungsbildung Richtung
  Full-Loss.
- **Excluded Loss:** Das Gegenteil — *entferne* alle Schlüsselfrequenzen, behalte
  den Rest. Das zerstört die Schaltung. Früh (Memorisierung) nutzt das Netz die
  Schlüsselfrequenzen noch nicht, also bleibt Excluded Loss niedrig; wenn die
  Schaltung entsteht, **steigt Excluded Loss stark an**.

Das Auseinanderlaufen dieser beiden Kurven ist die mechanistische Signatur der
Schaltungsbildung — und es zeigt die drei Phasen *ohne* die Test-Accuracy. Wir
prüfen die Korrektheit mit Sanity-Checks (Rücktransformation auf ~1e-13 genau;
„alle Frequenzen behalten" = Identität). Konkrete verifizierte Zahlen aus unserem
Lauf trägt RESULTS.md §2; den genauen Verlauf siehst du in
`figures/<id>_progress.png`.

---

## 6. Das eigene Experiment — Cross-Architektur (Erkenntnisgewinn)

**Frage:** Grokken verschiedene Architekturen modulare Addition gleich?

**Aufbau:** 1-Layer-Transformer vs. 2-Layer-MLP. Beide bekommen eine *geteilte*
Embedding-Tabelle `W_E[p, d]`, damit **dieselbe** Fourier-Analyse auf beide
anwendbar ist (sonst wäre der Vergleich unfair). Sonst alles gleich: `p=113`,
`wd=1.0`, `frac=0.5`, Grokfast, **3 Seeds je Architektur**, alle grokken.

**Ergebnis (Mittel ± Std über 3 Seeds):**

| | Generalisierungs-Schritt | top-8 Freq-Power | finale Test-Acc |
|---|---|---|---|
| **Transformer** | **750 ± 79** | **0.59 ± 0.05** (sparser) | 0.986 ± 0.003 |
| **MLP** | **2163 ± 59** | **0.35 ± 0.003** (verteilter) | 0.989 ± 0.004 |

**Befund:** Beide grokken zu ~99 %, aber der **Transformer grokt ~2.9× schneller
*und* mit deutlich sparserer/periodischerer Repräsentation**. Die Lücke (750 vs
2163) ist viel größer als die Seed-Streuung (≤80) → robust, kein Rauschen.
*Interpretation:* Der induktive Bias der Attention (`=` liest beide Operanden,
§5.4) führt den Transformer zu einer saubereren Trig-Schaltung; das MLP löst
dieselbe Aufgabe mit verteilteren Frequenzen.

**Ehrliche Grenzen (musst du selbst nennen, bevor die Jury es tut):**
- Ein **einzelner Hyperparameter-Punkt**, kein Sweep.
- **Grokfast-beschleunigt** (für beide identisch → fairer Vergleich, aber nicht
  die un-beschleunigte Einstellung).
- Die geringere MLP-Sparsity ist **teils architektonisch** (konkatenierte
  Operanden-Embeddings statt attention-kombiniert), nicht rein eine
  Grokking-Qualitätsfrage.

---

## 7. Grokfast — und die Ehrlichkeitsgrenze

**Was es ist** (Lee et al. 2024, arXiv:2405.20233): Grokking wird von den
*langsam variierenden* Komponenten des Gradienten getrieben. Grokfast hält einen
gleitenden Durchschnitt (EMA) des Gradienten und **verstärkt** diese langsame
Komponente. Effekt: der gleiche Phasenübergang passiert in viel weniger Schritten.

**Warum wir es benutzen:** Wir trainieren CPU-only (kein GPU). Ein un-beschleunigter
Lauf braucht ~8000+ Schritte (~24 min); für die *Seed-Robustheit* und das
Cross-Arch-Experiment (6 Läufe) wäre das viel Rechenzeit. Grokfast komprimiert die
**Zeitachse** desselben Phänomens.

**Die Ehrlichkeitsgrenze (kritisch!):** Grokfast *beschleunigt* Grokking, es
*erzeugt* es nicht. Deshalb reproduzieren wir den **un-beschleunigten** Lauf
zuerst (das ist die Referenz, der 8222-Schritte-Grok in §1) und nutzen Grokfast
nur für schnelle Iteration. Beide zeigen dieselbe Signatur: perfekte
Memorisierung → langes Plateau → scharfer Sprung. **Wenn die Jury fragt „ist das
mit Grokfast nicht geschummelt?": Nein — der echte un-beschleunigte 8222-Schritte-
Grok ist unser Headline-Resultat; Grokfast ist nur das Stoppuhr-Tempo für die
Nebenläufe, und das ist offengelegt.**

---

## 8. Ehrlichkeit & Reproduzierbarkeit (die DNA des Projekts)

- **Determinismus-Test:** Gleiche Seed ⇒ bit-identische erste-Batch-Logit-Summe
  (`2449.995445449221`). Beweist: gleiche Config + Seed → gleiche Kurve.
- **11/11 Korrektheits-Checks** (`test_core.py`): u. a. Fourier-Orthonormalität
  und **Recovery einer künstlich injizierten Frequenz** (wir bauen ein Signal mit
  bekannter Frequenz und prüfen, dass unsere Analyse sie zurückfindet — so wissen
  wir, dass die Fourier-Analyse *korrekt* ist, nicht nur plausibel).
- **Jede Zahl traceback:** Jeder Wert in UI und RESULTS stammt aus einem
  geseedeten Lauf in `runs/`, regenerierbar mit einem Befehl.
- **Negative/fragile Resultate werden berichtet**, nicht versteckt (RESULTS §5).

> Das ist nicht nur „nett" — es *ist* die These des Projekts. Ein
> Interpretierbarkeitsprojekt, das schummelt, widerlegt sich selbst.

---

## 9. Der Web-Explorer + LiveLab

- **Stack:** Next.js + React Three Fiber (3D im Browser), TypeScript. Lädt die
  *echten* exportierten Lauf-Daten (`web/public/data/`), keine erfundenen Werte.
- **Was man tut:** Einen Lauf wählen, die Zeitleiste **scrubben**, zusehen, wie
  die 113 Token-Embeddings vom Blob in den Ring wandern, dazu die Loss/Accuracy-
  Kurven mit Übergangsmarker und das Fourier-Spektrum — alles synchron zum Schritt.
- **LiveLab:** Ein winziges MLP trainiert **im Browser** (TensorFlow.js) auf
  `(a+b) mod 23`. Per Slider (Weight Decay, Train-Fraction) löst der Nutzer
  Grokking *selbst* aus. Verifiziert: grokt headless (Test 0.912). Das macht das
  Konzept *interaktiv begreifbar* — der stärkste „Wow"-Hebel für die Jury.

---

## 10. Einordnung in die Literatur (must-know Namen)

- **Power et al. 2022** (arXiv:2201.02177) — erste Beschreibung von Grokking.
- **Nanda et al. 2023** (arXiv:2301.05217, ICLR) — unser **Hauptziel**: der
  1-Layer-Transformer, die Fourier/Trig-Schaltung, restricted/excluded loss.
- **Liu et al. 2022, Omnigrok** (arXiv:2210.01117) — Grokking via Gewichtsnorm.
- **Lee et al. 2024, Grokfast** (arXiv:2405.20233) — Beschleunigung.

Du solltest in einem Satz sagen können, was jedes Paper beiträgt und was *wir*
davon reproduzieren vs. neu machen.

---

## 11. Schnell-Erklärungen (übe diese laut)

**Grokking in 30 s:** siehe §1-Kasten (Schüler-Analogie).

**Die Trig-Schaltung in 2 min:** „Das Netz stellt jede Zahl als Punkt auf einem
Kreis dar — als Winkel. Addieren modulo p ist dann Winkel-Addieren. Mit den
Additionstheoremen aus der Schule kann das Netz den Summenwinkel aus den
Einzelwinkeln ausrechnen, und am Ende gewinnt durch konstruktive Interferenz
genau die richtige Antwort. Wir *messen*, dass das Netz das tut: 76 % seiner
Energie sitzt in wenigen Frequenzen, die Embeddings liegen auf einem Ring, und die
Attention zieht beide Operanden gleich heran."

**Warum preiswürdig (nicht „nur Reproduktion"):** „Wir reproduzieren ehrlich und
un-beschleunigt, implementieren die *namensgebenden* Progress-Measures des Papers,
liefern einen seed-robusten eigenen Cross-Architektur-Befund **und** machen das
Ganze als interaktiven 3D-Explorer für jeden begreifbar — Wissenschaft, die man
anfassen kann, mit voller Ehrlichkeitsspur."

---

## 12. Was du als Nächstes selbst tun solltest (siehe EIGENLEISTUNG.md)

Dieses Dokument gibt dir das *Verständnis*. Echte, dokumentierte **Eigenleistung**
baust du, indem du Teile selbst machst und nachrechnest — die konkrete Liste steht
in `EIGENLEISTUNG.md`. Mindestens: die Trig-Herleitung selbst auf Papier, einen
Lauf selbst starten und die Zahlen selbst aus `run.json` verifizieren, und eine
eigene kleine Erweiterung (z. B. add-vs-mul) selbst formulieren.
