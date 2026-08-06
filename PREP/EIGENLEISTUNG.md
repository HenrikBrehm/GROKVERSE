# EIGENLEISTUNG — ehrlicher Aktionsplan (interne Vorbereitung, NICHT einreichen)

> Das ist das wichtigste strategische Dokument für deine BWKI-Chancen. Es geht
> **nicht** darum, Eigenleistung vorzutäuschen — das würde gegen die Regeln
> verstoßen und gegen die Ehrlichkeits-These des Projekts. Es geht darum, *echte*,
> nachweisbare Eigenleistung aufzubauen und sie so zu dokumentieren, dass die Jury
> sie klar erkennt. Liegt in `bwki/PREP/`, gehört **nicht** ins eingereichte Repo.

---

## 1. Das Problem, schonungslos

Die BWKI-Teilnahmebedingungen verlangen wörtlich:

> „Bei allen eingereichten Projekten muss ein **klar erkennbarer und nachvollziehbar
> dokumentierter Anteil an Eigenleistung** vorliegen."

Und **Eigenständigkeit ist das erstgenannte Bewertungskriterium**. Gleichzeitig
sagt deine (vorbildlich ehrliche) `AI_DISCLOSURE.md` aktuell sinngemäß: Code,
Experiment-Design, Interpretation und Doku sind KI-generiert; der Mensch hat
*dirigiert, reviewt, freigegeben, Zahlen verifiziert*. 

Das ist ehrlich — aber **„dirigieren und freigeben" ist Projektleitung, nicht der
technisch-wissenschaftliche Eigenanteil, den die Jury sucht.** Die ML-Forscher in
der Jury werden genau hier bohren. Wenn der einzige nachweisbare Eigenanteil das
Schreiben einer Constitution ist, ist das Projekt beim wichtigsten Kriterium
angreifbar — egal wie stark der Rest ist.

**KI-Nutzung ist erlaubt** (und du legst sie korrekt offen). Du wirst nicht
disqualifiziert, weil du Claude Code benutzt hast. Das Risiko ist rein die
**Bewertung der Eigenleistung** und die **Verteidigbarkeit im Finale**.

---

## 2. Die zwei Hebel (beide brauchst du)

**Hebel A — Verstehen (sofort, am wichtigsten).** Eigenleistung wird im Finale am
eigenen Stand und im Jury-Gespräch geprüft. Wenn du jede Zahl, jede
Designentscheidung und vor allem die **Trig-Schaltungs-Herleitung** selbst am
Whiteboard erklären und verteidigen kannst, wird der „KI-gebaut"-Einwand stumpf.
Verständnis schlägt Autorschaft. → Arbeite `STUDY_GUIDE.md` und `JURY_QA.md` durch,
bis du sie nicht mehr brauchst.

**Hebel B — Selbst machen & dokumentieren (über die nächsten Wochen).** Bau einen
echten, abgegrenzten Eigenanteil, den du nachvollziehbar dokumentierst. Du musst
nicht alles neu schreiben — aber es muss substanzielle, *deine* Beiträge geben, die
über „Review" hinausgehen. Konkrete Liste unten.

---

## 3. Konkrete Eigenleistungs-Bausteine (wähle mehrere, dokumentiere jeden)

Für **jeden** Baustein gilt: mach es selbst, und halte fest *wie* (Notiz, Skizze,
Commit von deiner Hand, kurzer Lab-Eintrag mit Datum). Das ist die „nachvollziehbare
Dokumentation", die die Regel fordert.

### Mathematik / Verständnis (am höchsten gewichtet, am wenigsten Aufwand)
- [ ] **Die Trig-Identitäts-Herleitung von Hand** (STUDY_GUIDE §5.2) auf Papier
  nachvollziehen und sauber abschreiben — Embedding-als-Winkel → MLP-Produkte →
  Additionstheorem → konstruktive Interferenz. *Das ist der intellektuelle Kern.*
- [ ] Eine **eigene Skizze/Erklärgrafik** zeichnen (z. B. der Ring + konstruktive
  Interferenz), die NICHT aus dem Code kommt. Eigene Visualisierung = Eigenleistung.
- [ ] Schriftlich (eigene Worte) beantworten: *Warum* verursacht Weight Decay
  Grokking? *Warum* p prim? *Warum* kein LayerNorm? *Warum* train_frac=0.3?

### Experimente selbst fahren & auswerten
- [ ] Einen Lauf **selbst starten** (`python -m grokverse.train ...`) und die Zahlen
  **selbst** aus `run.json` ablesen und gegen RESULTS verifizieren. Schreib auf,
  was du gemessen hast.
- [ ] **Eine eigene Erweiterung formulieren UND durchführen.** Stärkster Kandidat:
  das **add-vs-mul-Experiment** (grokt Multiplikation `(a·b) mod p` anders als
  Addition? Welche Struktur findet sie?) — die Daten-Pipeline unterstützt `mul`
  bereits (`data.py`). Das ist *dein* Experiment: du stellst die Frage, fährst die
  Läufe, wertest die Fourier-Struktur aus, schreibst den Befund. Selbst wenn KI dir
  beim Code hilft — die *wissenschaftliche Frage, Durchführung und Interpretation*
  sind deine.
- [ ] Alternative: den **un-beschleunigten Cross-Arch-Vergleich** bei frac=0.3
  selbst anstoßen (die in RESULTS genannte offene Grenze schließen) und auswerten.

### Eigene Analyse / Code
- [ ] Eine **eigene Mess- oder Plot-Funktion** schreiben (auch klein), die du
  vollständig verstehst — z. B. die Gewichtsnorm über die Zeit plotten (zeigt den
  Omnigrok-Mechanismus direkt!) oder die Verteilung der dominanten Frequenzen über
  Seeds. Klein, aber zu 100 % deins und erklärbar.
- [ ] Den `test_core.py`-Suite um **einen eigenen Test** ergänzen, den du dir
  ausgedacht hast.

### Dokumentation in deiner Stimme
- [ ] Die **schriftliche Projektdokumentation** (das BWKI-Einreichungsdokument) in
  *deinen* Worten schreiben — nicht die KI-Drafts 1:1. Die Jury merkt den
  Unterschied zwischen „verstanden und erklärt" und „generiert".
- [ ] Den **Video-Pitch selbst sprechen** und gestalten (das ist ohnehin Pflicht,
  3–5 min). Dein Gesicht, deine Stimme, deine Erklärung = sichtbare Eigenleistung.

---

## 4. Wie du es ehrlich dokumentierst (nicht schönfärben)

In der `AI_DISCLOSURE.md` (siehe die überarbeitete Version im Repo) und in der
Einreichungs-Doku:

- **Sei spezifisch statt vage.** Nicht „Mensch hat reviewt", sondern: „Henrik hat
  die Trig-Identitäts-Herleitung eigenständig nachvollzogen (Anhang X), das
  add-vs-mul-Experiment konzipiert, durchgeführt und ausgewertet (Abschnitt Y), die
  Gewichtsnorm-Analyse selbst implementiert (Datei Z), und die gesamte
  Einreichungsdoku + den Pitch selbst verfasst."
- **Trenne sauber:** eine ehrliche Tabelle „von KI generiert" vs. „von Henrik selbst
  erstellt/durchgeführt/verstanden". Je konkreter die rechte Spalte, desto besser.
- **Frame die KI-Nutzung als Methode, nicht als Ausrede.** „Ich habe einen
  autonomen Agenten unter einer strikten Anti-Fabrikations-Constitution
  *orchestriert* und jede wissenschaftliche Aussage selbst gegen die geloggten
  Kurven verifiziert" ist eine legitime, sogar interessante Arbeitsweise — *wenn*
  ein echter Eigenanteil (§3) dahintersteht und du alles verstehst.

> Faustregel: Für jede Aussage in deiner Doku solltest du im Jury-Gespräch sagen
> können „das habe ich so gemacht/gemessen/hergeleitet, und hier ist warum". Wo das
> (noch) nicht stimmt, ist es ein To-do aus §3, kein Formulierungsproblem.

---

## 5. Ehrliche Einschätzung der Wirkung

- Machst du **Hebel A** (verstehen) gründlich und **2–3 Bausteine aus §3** selbst:
  Der Eigenständigkeits-Einwand verliert seine Schärfe; das Projekt spielt seine
  echten Stärken (Schwierigkeitsgrad, wiss. Arbeiten, Artefakt) aus → realistischer
  Finalist (Top 10).
- Machst du zusätzlich ein **eigenes Experiment** (add-vs-mul) mit eigener
  Auswertung: Du hast einen klar abgegrenzten, vorzeigbaren Erkenntnisgewinn, der
  *unzweifelhaft* deiner ist → der Hauptpreis wird denkbar.
- Lässt du es wie es ist (rein KI-gebaut, „nur reviewt"): technisch top, aber beim
  erstgewichteten Kriterium angreifbar — und das kostet im Zweifel die Spitze.

**Die gute Nachricht:** Einsendeschluss ist Ende September. Du hast Zeit für §3.
Fang mit der Trig-Herleitung (eine Stunde) und dem add-vs-mul-Experiment an — das
sind die zwei mit dem besten Verhältnis aus Aufwand und Eigenständigkeits-Gewinn.
