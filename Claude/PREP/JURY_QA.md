<!-- INTERNES VORBEREITUNGSMATERIAL - NICHT EINREICHEN. Liegt in bwki/PREP/. -->
<!-- Erzeugt aus einem 5-Personen-Jury-Panel (mechanistic-interp, ML-Theorie, Reproduzierbarkeit,
     Eigenstaendigkeit/KI-Nutzung, Wissenschaftskommunikation) + Synthese. Antworten von Henrik
     gegenzulesen, zu verstehen und in eigene Worte zu fassen - nicht auswendig vortragen. -->

# GROKVERSE - BWKI-Jury-Verteidigung: Vorbereitungsdokument

> Stand: 21.06.2026 | Projekt: Reproduktion + mechanistische Analyse von Grokking (Nanda et al. 2023, arXiv:2301.05217) plus eigenes Cross-Architektur-Experiment.
> Leitprinzip dieser Verteidigung: **Ehrlich, faktenverankert, Grenzen offen benennen.** Jede Zahl traced zu einem geseedeten Lauf in `runs/`. Nie etwas behaupten, das nicht gemessen ist.

---

## 0. KILLER-FRAGEN, die du SICHER beherrschen MUSST

Diese acht entscheiden die Verteidigung. Wenn du nur diese sitzt hast, bist du tragfaehig.

1. **Eigenleistung: Was hast DU getan, nicht der Agent?** -> Problemdefinition + Anti-Fabrikations-Constitution + Experimentauswahl (A statt Fallback B) + faktentreue Verifikation gegen geloggte Kurven (inkl. Erkennen der Grokfast-vs-Referenz-Diskrepanz) + KI-Offenlegung. Code-Anteil offen als KI deklariert. (Siehe Block 7)
2. **Herleitung restricted/excluded loss aus dem Gedaechtnis.** -> 2D-Fourier ueber (a,b)-Gitter, Schluesselfrequenzen behalten = restricted (faellt Richtung full), entfernen = excluded (steigt). Gegenlaeufigkeit ist der Befund. (Siehe Block 2 + 7)
3. **Kausalitaet vs. Korrelation der Trig-Identitaet.** -> Ehrlich: korrelativ belegt, nicht voll kausal. Was fehlt klar benennen. (Siehe Block 2)
4. **Zirkularitaet der Progress Measures.** -> Aussagekraft kommt aus zeitlicher ENTKOPPLUNG (restricted faellt / excluded steigt), nicht aus der per-Konstruktion-wahren End-Wichtigkeit. (Siehe Block 3)
5. **Statistik n=3 + "robust".** -> Effektgroesse vs. Streuung: 1413 Schritte Luecke = ~18 Seed-std; nichtparametrisch, keine Praezisionsstatistik behauptet. (Siehe Block 6)
6. **Grokfast-Konfundierung des 2.9x-Befunds.** -> Un-beschleunigter MLP-Lauf FEHLT. Claim eng: "unter Grokfast". Nicht extrapoliert. (Siehe Block 5)
7. **Single-Hyperparameter-Punkt / kein Sweep.** -> Starke LOKALE, schwache GLOBALE Aussage. Bewusst nur lokal verkauft. (Siehe Block 5 + 6)
8. **Praktische Relevanz "KI-Sicherheit" - Marketing?** -> Bezug ist konzeptionell, nicht praktisch. Grokking als sauberes Interpretierbarkeits-Labor; from-inside-Masse. Skalierung offen. (Siehe Block 8)

---

## Block 1: Grokking-Grundlagen & kanonische Reproduktion

### Was ist Grokking (Laien-Pitch, 60 Sek., ohne Fachwoerter)
Ein kleines Netz bekommt eine simple Mathe-Aufgabe ("a plus b, dann Rest bei Teilung durch 113"). Anfangs lernt es nur auswendig: 100% auf Bekanntem (nach 145 Schritten), Zufallsniveau auf Neuem. Dann passiert lange scheinbar nichts (~8000 Schritte). Ploetzlich, bei Schritt 8367, "versteht" es die Regel und loest auch nie gesehene Aufgaben (98%). Dieser Sprung vom Auswendiglernen zum Verstehen heisst Grokking. Relevanz: Ein Modell kann lange schlecht aussehen und dann sprunghaft eine echte Faehigkeit ausbilden, die man von aussen kaum erkennt. Ehrliche Grenze: Spielzeugbeispiel, kein grosses Sprachmodell.

### Kanonische Zahlen (alle aus dem un-beschleunigten Referenzlauf, frac=0.3, seed0)
- Train acc 1.0 bei **Schritt 145** (Memorisierung)
- Test acc kreuzt 0.95 bei **Schritt 8367**, finale test acc **0.981**
- **Grok-Gap 8222 Schritte**
- Test-Loss steigt im Plateau auf **~26**, faellt dann auf **0.06**

### Warum Test-Loss auf ~26 (weit ueber Zufall ln(113)~4.7)?
Kein Bug, sondern erwartetes Verhalten: Unter hohem weight decay (1.0) wird das Netz auf Trainingsdaten zunehmend konfident -> hochkonfident FALSCHE Vorhersagen auf Test -> Cross-Entropy bestraft konfident-falsch exponentiell. Dokumentiertes Grokking-Muster (vgl. Liu et al. Omnigrok zur Loss/Norm-Dynamik). Abgesichert: Reconstruction-Sanity ~1e-13 (numerisch korrekt) + finaler Loss konvergiert sauber auf 0.06. **Grenze:** Spike nur fuer diesen einen Lauf charakterisiert, nicht ueber Seeds gemittelt.

### WICHTIG: Single-Seed der kanonischen Zahl
Die Zahl 8367 / Gap 8222 ist ein **Single-Seed-Resultat** (txf_add_p113_wd1.0_frac0.3_seed0). Ich verkaufe sie NICHT als Verteilung, sondern als **Existenz- und Phasen-Nachweis**: bit-deterministisch, vollstaendig zu `runs/` ruecktracebar, Phasenstruktur konsistent mit Literatur. Quantitative Verteilungsaussagen mache ich nur dort, wo ich Seeds habe (Cross-Arch). Seed-Sweep der Referenz fehlt aus CPU-Laufzeitgruenden - echte Grenze, nicht kaschiert.

### Warum weight_decay=1.0 (extrem hoch) und kein LayerNorm?
- **wd=1.0:** Grokking ist eng an die Gewichtsnorm gekoppelt (Liu et al., Omnigrok, arXiv:2210.01117). Hohes wd treibt das Netz nach der Memorisierung aktiv von der grossen-Norm-Memorierungsloesung weg zur kleinen-Norm-generalisierenden Loesung. Der Regularisierungsdruck ist der Motor, der verzoegerte Generalisierung erzwingt.
- **kein LayerNorm:** haelt die mechanistische Analyse sauber. Ohne LN ist Embedding->Logits naeher an linearen/bilinearen Operationen, sodass die Fourier-Zerlegung die Trig-Schaltung direkt sichtbar macht. LN fuehrt eingabeabhaengige Reskalierung ein, die die Frequenzinterpretation verschmiert.
- **Ehrliche Grenze:** Rezept methodentreu von Nanda uebernommen (so deklariert). Kein eigener Ablations-Sweep (mit/ohne LN, wd-Variation) in DIESEM Repo - stuetze mich auf Literatur + einen reproduzierten Lauf. **[VORBEREITEN: Ablations-Sweep wd-Reihe + mit/ohne LN als erster Nachfolge-Schritt skizzieren koennen.]**

### Warum train_frac kausal entscheidend ist (0.3 vs 0.5)
train_frac ist KEIN Laufzeit-Detail, sondern der zentrale Kontrollhebel: Bei kleiner Fraktion gibt es viele memorisierende Loesungen, aber nur wenige generalisierende -> Netz findet zuerst Memorisierung, erst unter wd-Druck die sparse trig-Schaltung -> grosse Luecke. Erhoeht man train_frac, schrumpft die Memorisierungs-Nische, Luecke wird kleiner/Grokking frueher; unter einer kritischen Fraktion grokt es gar nicht. Deshalb: **frac=0.3 = kanonische Reproduktion** (Headline-Luecke), **frac=0.5 nur im Cross-Arch-Vergleich** (beide Architekturen identisch). Zwei verschiedene Betriebspunkte, in RESULTS.md getrennt - KEINE durchgaengige Kurve.

---

## Block 2: Mechanistische Analyse - Fourier, Trig-Identitaet, Attention

### Die Schaltung (Wow-Moment, Laien-tauglich)
Traegt man die gelernten Zahlen-Darstellungen geometrisch auf, ordnen sie sich auf einem **Kreis** an. Das Netz hat selbst entdeckt: modulare Addition = Drehen auf einem Ziffernblatt (Zahlen addieren = Winkel addieren). **Ehrlich:** Diese Kreis-Interpretation ist NICHT meine Entdeckung, sondern reproduziert Nanda et al. 2023. Mein Beitrag: sie nachvollziehbar, interaktiv und mit ruecktracebaren Zahlen erfahrbar machen.

### Schluesselfrequenzen & Sparsity
- Gegrokter Lauf: top-8 Fourier-Frequenzen halten **~76%** der Power; nicht-gegrokter Lauf nur **~32%** (diffus).
- Konkrete k im Referenzlauf: **18, 15, 1, 11, 13, 22, 56, 36**.
- Cross-Arch (ueber Seeds): Transformer top-8-Power **0.59 +/- 0.05** vs MLP **0.35 +/- 0.003**.

### Kausalitaet vs. Korrelation der Trig-Identitaet (KILLER #3)
**Ehrlich: Ich belege die Trig-Identitaet cos(a)cos(b)-sin(a)sin(b)=cos(a+b) KORRELATIV, nicht voll kausal.**
Was ich habe: (a) sparse-periodisches Embedding-Spektrum (top-8 ~76%); (b) finale Logits im 2D-Fourier ueber (a,b) auf wenige Schluesselfrequenzen konzentriert (sonst koennte restricted full nicht erreichen; recon-sanity ~1e-13 garantiert echten Effekt, kein Transform-Artefakt); (c) Attention 50/50 auf beide Operanden. Konsistent mit der Identitaet, aber **kein Beweis, dass das MLP exakt cos(a)cos(b)-sin(a)sin(b) rechnet** - ein Netz koennte periodische Logits ueber dem Gitter erzeugen, ohne die Identitaet intern zu implementieren.
**Was fuer Kausalitaet fehlt** (offen benennen): (1) Logit-Frequenzspektrum gegen Embedding-Spektrum matchen (aus Daten ziehbar, nicht publiziert); (2) Neuron-Aktivierungen gegen cos/sin(w_k(a+b)) fitten, R^2 zeigen; (3) Gewichts-Ablation einzelner Frequenzrichtungen mit Accuracy-Drop. Mein excluded loss ist eine LOGIT-seitige Ablation (erster kausaler Schritt), operiert aber auf Output-Logits, nicht internen Gewichten - belegt "die Schaltung NUTZT diese Frequenzen", nicht "sie implementiert die Identitaet so".

### Warum genau diese 8 Frequenzen? / Zahlentheorie
Die spezifische Menge ist die **freie Wahl des Netzes** (RESULTS: sparse Struktur robust, exakte Frequenzen seed-abhaengig). Fakten: Fuer (a+b) mod p funktioniert die Trig-Identitaet fuer JEDES k - keine ausgezeichnete Frequenz; das Netz waehlt eine ausreichend grosse Teilmenge, die die Logits ueber dem Gitter aufloest. k=2,3 "fehlen" nicht als Defekt - das Netz hat sie nicht gewaehlt. k und p-k sind in der reellen Basis nicht separat: ich indexiere k nur bis (p-1)/2=56, jede Frequenz ist ein cos/sin-Paar (beide Vorzeichen), k=56 ist legitim. **[VORBEREITEN: Keine Erklaerung, WARUM gerade diese 8 - seedspezifisch, nach Forschungsstand nicht deterministisch erwartbar. Diesen Satz souveraen sagen koennen.]**

### Attention 50/50 - Einwand "ist nur der Nullpunkt"
50/50 allein ist schwach (Gleichverteilung ist der Softmax-Nullpunkt). Was meinen Befund stark macht ist die **dritte Zahl**: Selbst-Attention des "="-Tokens ist **~0.001, praktisch NULL**, ueber alle 12.769 Eingaben und in jedem der 4 Heads (z.B. h0: 0.502/0.498/0.001). Ein uninformiertes Netz wuerde ueber 3 Positionen ~1/3 verteilen, inkl. Selbst-Attention. Dass "=" sich selbst fast komplett ignoriert und die Masse exakt auf beide Operanden legt, ist nicht der Nullpunkt.
**Ehrliche Grenze:** Nur am FINALEN Modell gemessen, nicht als Zeitreihe ueber das Plateau. Voller Schlag des Einwands braeuchte attention_pattern an den gespeicherten Zwischen-States (liegen in train_capturing_states vor) -> zeigen, dass Selbst-Attention waehrend Memorisierung hoeher ist und erst mit Generalisierung gegen 0 kollabiert. **[VORBEREITEN: Endzustands-Beleg, kein Verlaufs-Beleg - mit vorhandenem Code schliessbar.]**

---

## Block 3: restricted / excluded loss - Definition, Validitaet, Zirkularitaet

### Herleitung (KILLER #2 - MUSST du ohne Notizen koennen)
Die Logits am "="-Token sind eine Funktion von (a,b) auf dem p×p-Gitter (p²=12.769 Werte pro Klasse). Man transformiert diese Funktion in die 2D-Fourier-Basis ueber (a,b).
- **restricted loss:** Logits NUR aus den Schluesselfrequenz-Komponenten rekonstruieren, Loss berechnen. Faellt er Richtung full loss -> die Schaltung steckt fast vollstaendig in diesen Frequenzen.
- **excluded loss:** genau diese Frequenzen entfernen, aus dem Rest rekonstruieren. Steigt er stark -> ausserhalb der Schluesselfrequenzen keine nutzbare Information mehr (Memorierungsrauschen weggeraeumt).
- **Kern:** Beide Masse brauchen KEINEN held-out-Split - sie messen Fortschritt direkt an der Trainingsfunktion (auf vollem (a,b)-Gitter, alle p*p Eingaben, all_y). Deshalb sind sie definitionsgemaess keine Funktion der Test-Accuracy.

### Zirkularitaet / Selbstbestaetigung (KILLER #4)
Vorwurf: Frequenzen aus Endzustand fixiert, retroaktiv angewandt -> zirkulaer/Tautologie.
**Entkraeftung:** Aussagekraft kommt NICHT daraus, dass die Frequenzen am Ende wichtig sind (per Konstruktion wahr), sondern aus der **zeitlichen ENTKOPPLUNG**:
- Trivial waere, wenn NUR restricted faellt. Nicht-trivial: **excluded steigt gleichzeitig stark.**
- Frueh (Memorisierung): excluded NIEDRIG - das Netz loest Trainingsdaten OHNE die spaeteren Frequenzen; entfernt man sie, bleibt die memorierte Loesung intakt.
- Spaet: excluded steigt scharf - jetzt haengt die Loesung an genau diesen Frequenzen. Das ist durch die Projektion nicht erzwingbar; es misst kausale Abhaengigkeit.
- Die EINMAL-final-Fixierung (identisch auf alle Schritte) ist Absicht: fairer Zeitvergleich ohne per-Schritt-Nachtunen.
**Ehrliche Grenze:** Das Mass setzt die Hypothese "periodische Schaltung" voraus und kann eine voellig andere Loesungsstruktur nicht detektieren - es ist ein Fortschrittsmass FUER diese Schaltung, kein architektur-agnostischer Generalisierungs-Detektor. Beweist NICHT, dass die Frequenzen die einzige/kausal hinreichende Schaltung sind.

### Reagiert excluded loss VOR der test acc? (KILLER-naher Punkt)
Das ist der eigentliche Witz eines Progress Measures bei Nanda (misst Schaltungsbildung waehrend des Plateaus, bevor test acc sich bewegt).
**Was ich belegen kann:** Konstruktion ist held-out-UNABHAENGIG (volles Gitter, nicht Test-Split). **Was ich noch NICHT als Headline-Zahl habe:** den ersten Schritt, an dem excluded signifikant ueber Baseline steigt, gegen transition.test_generalized_step (Schritt 8367). Die Daten liegen in der JSON (measured_steps, excluded_loss, test_acc_at_measured nebeneinander), aber den Vorlauf habe ich nicht extrahiert - **und erfinde ihn nicht.**
**Ehrlich:** Mein staerkster belegter Claim ist, dass restricted/excluded die memorize->circuit-Transition held-out-unabhaengig nachzeichnen. Der schaerfere Nanda-Claim "reagiert N Schritte VOR test acc" steht und faellt mit dieser Zahl. Caveat: logarithmisches Logging + Subsample auf 120 Punkte begrenzt die Zeitaufloesung im Plateau. **[VORBEREITEN: Den Vorlauf in Schritten aus der JSON extrahieren - wenn vor der Verteidigung machbar, ist das ein Headline-Upgrade. Sonst Grenze offen sagen.]**

### Restricted-Loss-Definition: warum +0.10-Toleranz?
Check akzeptiert restr_final < full_final + 0.10.
- Die +0.10 ist eine **pragmatische Schwelle fuer den boolean-Check**, KEIN gemessener Endwert. Der echte Wert ist final_restricted_loss minus final_full_loss in der JSON - nur DER zaehlt.
- Logik: restr < full + epsilon prueft "die Schluesselfrequenzen REICHEN"; epsilon>0 erlaubt minimale Unvollstaendigkeit, ohne den Check als Bug zu werten.
- **Berechtigte Konsequenz:** Verfehlt restricted full um eine spuerbare Marge, traegt mindestens eine nicht-erfasste Frequenz noch Signal -> verweist auf die max_k=8-Deckelung (s.u.). NICHT behaupten darf ich, dass restricted full "unterbietet" (Cleanup-Effekt) - das waere ein staerkerer Nanda-Befund, nur am Langlauf zeigbar (nicht getan). **[VORBEREITEN: exakten restr_final - full_final-Wert aus JSON parat haben.]**

### Schluesselfrequenz-Selektion: max_k=8 (KILLER-nah)
dominant_frequencies waehlt ueber 90%-kumulativer-Power-Threshold, gedeckelt bei max_k=8. **Ehrlich: max_k=8 ist eine gesetzte Obergrenze, kein gemessenes Faktum - eine Grenze gegenueber Nanda (bei dem die Frequenzanzahl ein gemessenes Faktum ist).**
Verteidigung: (1) Schaltung robust dominant - top-8 ~76% (grokkt) vs ~32% (nicht-grokkt), Signal nicht grenzwertig. (2) excluded loss loescht jede 2D-Komponente, die EINE Schluesselfrequenz beruehrt; nimmt man eine echte Frequenz zu wenig mit, steigt excluded WENIGER -> Fehler ist konservativ, nicht ueberzeichnet. (3) top-8 ist Nanda-Konvention (kleine Zahl dominanter Frequenzen), kein post-hoc Tuning zur Kontrast-Maximierung.
**Was fehlt:** Ablationslauf ueber k in {5,6,8,10,12}, der zeigt, dass final_excluded_loss mit k waechst und ab dem wahren k ein Plateau erreicht. Sanity-Check zeigt nur dass die Menge AUSREICHT, nicht dass sie MINIMAL ist. **Minimalitaet habe ich nicht bewiesen.** **[VORBEREITEN: k-Sweep als sauberen Falsifikations-/Nachfolge-Schritt benennen.]**

---

## Block 4: Drei Phasen, Progress Measures als Diagnose

### Sehe ich Nandas drei Phasen (memorize, circuit formation, cleanup)?
- **Phase 1+2 belastbar:** Uebergang Memorisierung -> Schaltungsbildung sauber gezeigt (excluded frueh niedrig, spaet hoch).
- **Phase 3 (cleanup) nur schwach aufgeloest:** Ich breche bei early-stop-acc ab und subsample auf 120 Schritte. Saubere cleanup-Phase muesste am un-beschleunigten Langlauf ueber das Plateau hinaus gezeigt werden - **nicht voll ausgefahren.**
- **Ehrliche Bilanz: Zwei Phasen belastbar, die dritte nur andeutungsweise.** **[VORBEREITEN: Langlauf ueber Generalisierung hinaus als Nachfolge.]**

### Progress Measures als Diagnose, nicht Kausalbeweis
Ich verteidige die Masse NICHT als kausal, sondern als **Fortschritts-DIAGNOSE im Sinne von Nanda**: held-out-unabhaengiges Mass dafuer, dass genau die spaeter dominante Struktur progressiv aufgebaut wird - nicht mehr, nicht weniger.

---

## Block 5: Cross-Architektur-Experiment (Transformer vs MLP)

### Der Kernbefund
Unter identischen, fairen Bedingungen (p=113, wd=1.0, frac=0.5, Grokfast alpha=0.98/lambda=2.0, geteilte Embedding-Tabelle, 3 Seeds je Architektur):
- Transformer grokt bei **750 +/- 79** Schritten, MLP bei **2163 +/- 59** -> **~2.9x schneller**.
- top-8-Power: Transformer **0.59 +/- 0.05** vs MLP **0.35 +/- 0.003** (sparsere Darstellung).

### "Geteilte Embedding-Tabelle" - was heisst das?
Gleiche **Struktur/Anwendbarkeit der Fourier-Analyse**, NICHT geteilte Gewichte. Beide trainieren ihr eigenes W_E, deshalb divergieren die Spektren. Vorkehrung garantiert: derselbe Fourier-Apparat ist auf beide anwendbar, Power-Unterschied nicht durch unterschiedliche Embedding-Raeume erklaerbar.

### Misst top-8-Power beim MLP ueberhaupt die Schaltung? (KILLER)
**Schaerfster Einwand, gebe ihm grossteils recht.** Beim MLP ist das Embedding NICHT der Hauptort der Berechnung - das MLP konkateniert die Embeddings und rechnet in den dichten Schichten; der Transformer kombiniert ueber Attention. top-8-Power auf W_E misst beim Transformer naeher an der Schaltung als beim MLP. Genau deshalb in RESULTS: "geringere MLP-Sparsity teils architektonisch (konkatenierte Operanden vs attention-kombiniert)" - kein reiner Schaltungs-Qualitaets-Vergleich.

### Mess-Artefakt vs echter Effekt - wie viel ist Artefakt? (KILLER)
Ich kann die Beitraege "reine periodische Schaltung" vs "Konkatenations-Geometrie" an den 0.35 mit den vorhandenen Laeufen **NICHT sauber quantitativ trennen** - deshalb steht die Einschraenkung explizit drin. Was den Befund nicht wertlos macht: "teils architektonisch" ist kein Bug, sondern Teil der Erklaerung, WARUM sich die Architekturen unterscheiden. **Saubere Trennung braeuchte ein Kontroll-MLP mit additiver statt konkatenierter Eingabe (nicht gelaufen).** Robuster Kern bleibt die **Grokking-ZEIT-Luecke (750 vs 2163), die nicht von der Sparsity-Metrik abhaengt.** **[VORBEREITEN: Kontroll-MLP additiv als Nachfolge benennen.]**

### Konfundierung an dutzenden Achsen
**Korrekt und so deklariert.** Kontrolliert: gleiches p, wd, train_frac, geteilte Embedding-Struktur, identisches Grokfast. NICHT kontrolliert: Parameterzahl, Tiefe, Attention-vs-Konkatenation. Mein Anspruch ist eng: an diesem fairen, gleich beschleunigten Punkt grokt der Transformer schneller + sparser. Kausale Zerlegung (Attention? Parameterzahl? Tiefe?) leistet dieser Punkt nicht - braeuchte Sweep mit matched parameter count + ablatierter Attention (als naechster Schritt notiert). **[VORBEREITEN: matched-param-count + Attention-Ablation skizzieren.]**

### Regime-Mismatch: frac 0.3 -> 0.5 UND Grokfast gleichzeitig
Der 2.9x-Befund stuetzt sich NICHT auf die frac=0.3-Referenzzahl und braucht das nicht. Es ist eine reine **INNERHALB-Experiment-Relation** - beide Architekturen unter identischen Bedingungen, nur dort verglichen. Die frac=0.3-Reproduktion dient nur dem Existenz-/un-getrickst-Nachweis, bevor ich mit Grokfast beschleunige. **Ich uebertrage keine absolute Schrittzahl von einem Regime ins andere.** Bleibende Grenze: kein un-beschleunigter Cross-Arch-Vergleich, kein Vergleich bei gleichem frac zwischen Referenz und Experiment.

---

## Block 6: Statistik, Rigor, Messdefinition

### n=3 + "robust" (KILLER #5)
n=3 erlaubt keine Praezisionsstatistik - **das behaupte ich nicht.** Robustheitsargument ist bewusst **nicht parametrisch, auf Effektgroesse vs. Streuung gestuetzt:**
- Luecke 1413 Schritte = rund **18 Seed-Standardabweichungen**; Verteilungen ueberlappen nicht annaehernd; alle 3/3 Seeds jeder Architektur ordnen sich gleich.
- Bei perfekter Trennung 3 gegen 3 ist selbst ein nichtparametrischer Permutationstest am Signifikanz-Floor (kleinster p-Wert 1/C(6,3) = 0.05).
- n=3 ist hart an der Mindestgrenze, die PROMPT.md fordert (>=3 Seeds); fuer Publikation wuerde ich 5-10 rechnen.
- Was ich NICHT mache: p-Wert auf die 79-vs-59-std setzen oder enge Konfidenzintervalle behaupten. Aussage: nur Richtung + Groessenordnung der Luecke ist an diesem einen Hyperparameter-Punkt seed-stabil.

### Single-Hyperparameter-Punkt (KILLER #7)
Gesamtes quantitatives Hauptresultat = EIN Punkt (p=113, wd=1.0, frac=0.5). **Ich weiss nicht, ob der Effekt bei wd=0.5 oder frac=0.4 bleibt - behaupte es nicht.** Starke LOKALE Aussage (Luecke >25x Streuung -> kein Rauschen), schwache GLOBALE Aussage. Sweep ueber wd und frac fehlt (CPU-Budget). Fuer eine architektur-allgemeine Aussage ist das Resultat nicht hinreichend - zentrale Grenze des Eigen-Experiments.

### Messaufloesung / Logging-Raster
detect_transition nimmt den ersten geloggten Schritt mit test acc >= 0.95 UND train acc >= 0.99 -> berichtete Schrittzahl ist eine **OBERE Schranke** bis zum naechsten Log-Punkt, nicht der exakte Kreuzungsschritt. Bei logarithmischem ~150-Punkte-Raster ist relative Aufloesung naeherungsweise konstant (einstelliger Prozentbereich); fuer 8367 -> Intervallbreite Groessenordnung einige hundert Schritte. **Fairness:** Derselbe Mechanismus trifft Transformer UND MLP identisch; im Verhaeltnis hebt sich der systematische Anteil weitgehend heraus, und die Quantisierung ist viel kleiner als die 1413-Schritt-Luecke. Ich verkaufe keine Einer-Genauigkeit.

### Seed-Kopplung: Datensplit und Init an derselben Seed
In data.py wird perm via manual_seed(cfg.seed) gezogen, set_seed(cfg.seed) initialisiert das Modell - **beide an derselben Seed**. Pro Seed variiere ich BEIDES gemeinsam. Das ist fuer die Robustheitsaussage **kein Nachteil, eher staerker:** der Befund haelt ueber gemeinsame Variation beider Quellen. Was ich dadurch NICHT kann: Split-induzierte von Init-induzierter Streuung trennen (79-vs-59-std mischt beide). Trennung braeuchte zwei getrennte Generatoren (Split-Seed vs Init-Seed entkoppeln + kreuzen) - billiger Zusatzlauf. **[VORBEREITEN: entkoppelte Generatoren als Nachfolge benennen.]**

### Sanity-Checks: was ist trivial, was nicht?
- **Trivial (Fundament, kein Befund):** Reconstruction-Sanity ~1e-13 = numerische Korrektheit (Basis orthonormal, inv2d exaktes Inverse von fwd2d). Zweck **defensiv:** garantiert, dass ein excluded-loss-Anstieg kein Transform-Bug ist. Plus all_modes-Check (alle Moden = Identitaet, ~1e-13) prueft die Maskierungs-Logik.
- **Nicht-trivial (die wissenschaftlichen Aussagen):** (1) restricted_recovers_full (~8 von 56 Frequenzen tragen die ganze Loesung); (2) excluded_destroys_solution (Loss > 1.0 hoch); (3) Kontrast 32% vs 76% zum nicht-grokkten Lauf.
- Gesamt: **11/11 Korrektheits-Checks**, inkl. injizierter-Frequenz-Recovery (bekannte Frequenz einspeisen, exakte Wiedergewinnung pruefen) + Orthonormalitaet (F@F.T=I bis 1e-8) + Labels-Check ((a+b) mod p == y).

### Determinismus - was beweist er WIRKLICH?
- **Belegt:** gleiche Seed -> bit-identische erste-Batch-Logit-Summe **2449.995445449221** (Init + Forward deterministisch).
- **Zweck (eng, ehrlich):** Traceability-Garantie / Regressions-Wachhund gegen stille Nicht-Determinismus-Bugs (unkontrollierte Shuffles, ungeseedete Init), damit jede UI-Kurve exakt aus dem benannten run/ stammt.
- **Was es NICHT ist:** kein Plattform-Determinismus, keine Seed-Robustheit (das tut das 3-Seed-Experiment), keine wissenschaftliche Reproduzierbarkeit des Befunds. Wenn die Doku das anders suggeriert -> Framing-Fehler, korrigiere ich.

### Determinismus der Analyse-Pipeline (zwei getrennte Laeufe)
progress_measures.py macht ein EIGENES Re-Training (train_capturing_states), getrennt vom run.json-Lauf, aber mit set_seed(cfg.seed), gleicher AdamW-/Grokfast-Logik - Absicht: exakte Replik.
- **Belegt:** Init+Forward bit-identisch; rekonstruierte Transition (detect_transition) trifft die in run.json gespeicherte.
- **NICHT behaupten:** bit-Identitaet der GESAMTEN Kurve zwischen beiden Pipelines (train.py loggt pro Schritt nur Embeddings, progress_measures volle States; Snapshot-Reihenfolge koennte CPU-Reduktions-Reihenfolgen beruehren). **Vollstaendige bit-Identitaet ueber alle Schritte ist mein Anspruch, nicht mein bewiesener Messwert** - als offener Pruefpunkt deklariert.

### Cherry-Picking-Vorwurf (spaeterer Zeitstempel eines Laufs)
Aggregate gemeldet ueber GENAU drei Seeds (seed0/1/2 unter frac0.5_gf2.0), NICHT "die drei besten von vielen". Spaeterer Zeitstempel (10:35) erklaerbar: ein einzelner Lauf re-erzeugt (z.B. fuer Figuren); wegen bit-deterministischer Seeding ist ein re-run von seed0 per Konstruktion identisch zum Original - der Determinismus-Test ist genau der Schutz gegen "heimlich besseren Lauf". **Unabhaengig pruefbar:** seed0 selbst re-runnen, Logit-Summe + Endmetriken vergleichen. Keine verworfenen Zusatz-Seeds in runs/ - die Verzeichnisliste ist die Pruefspur.

---

## Block 7: Eigenstaendigkeit & KI-Nutzung

### Was hat HENRIK getan, nicht der Agent? (KILLER #1)
Code-Anteil offen als KI deklariert (AI_DISCLOSURE.md, Zeile-fuer-Zeile AI-vs-Mensch-Tabelle). Eigenleistung liegt in vier ueberpruefbaren Dingen:
1. **Constitution (PROMPT.md/PLAN.md):** Intent, Scope, Anti-Fabrikations-Regeln, Verifikationsdisziplin festgelegt und bindend gemacht - die wissenschaftliche Rahmensetzung.
2. **Experimentauswahl:** PLAN.md Phase 4 bot (A) Cross-Architektur und (B) Omnigrok-Fallback; ich habe bewusst (A) gewaehlt - das riskantere, erkenntnisreichere Experiment statt des signalsicheren Fallbacks. Plus Bedingung "muss ueber >=3 Seeds ueberleben, sonst Rauschen".
3. **Faktentreue Verifikation:** aktives Review - z.B. erkannt, dass eine Changelog-Notiz (179/675, Gap 496, Grokfast-Lauf) NICHT mit dem kanonischen un-beschleunigten Lauf (145/8367, Gap 8222) uebereinstimmt.
4. **Wissenschaftliche Urteilsrufe:** z.B. dass der un-beschleunigte Lauf die Referenz sein muss.

Ich behaupte NICHT, dass Review allein hohe Eigenleistung ist. Ich beanspruche: **Problemdefinition + Experimentwahl + faktentreue Verifikation** ist der Teil, fuer den ich geradestehe.

### Die zwei widerspruechlichen Zahlen (Gap 496 vs Gap 8222)
**Beide stimmen - fuer verschiedene Laeufe, und der Unterschied ist der Punkt.**
- Gap 496 (memorize 179, generalize 675) = **Grokfast-beschleunigter** Lauf (fuer CPU-Laufzeit).
- Gap 8222 (train 1.0 bei 145, test 0.95 bei 8367, finale 0.981) = **un-beschleunigter kanonischer** Referenzlauf (Nanda-Rezept).
PROMPT.md §7 verlangt explizit, das un-beschleunigte Phaenomen ZUERST zu reproduzieren. Dass die Changelog-Zeile den Grokfast-Lauf zeigt, ist ein **Doku-Schoenheitsfehler**, der korrigiert werden muss (Referenz im Writeup = un-beschleunigter Gap 8222). Dass ich diese Diskrepanz erkenne, IST der Verifikationsanteil, den ich als Eigenleistung beanspruche.

### Wissenschaftliche Konsistenz: "Masse nicht implementiert" vs Code implementiert sie
RESULTS.md §5 behauptet, restricted/excluded seien "nicht implementiert (brauchen per-step full-model checkpoints)". **Wahr ist: Sie SIND implementiert** - progress_measures.py loest das per-step-checkpoint-Problem durch Re-Training mit State-Capture statt nachtraeglichem Laden. **Die §5-Zeile ist VERALTET** (aus der Phase vor der Implementierung, nicht nachgezogen). Echter Dokumentationsfehler, muss vor Einreichung korrigiert werden. **Bedeutet NICHT, dass Zahlen unzuverlaessig sind** (jede Zahl traced zu geseedetem Lauf, recon-sanity ~1e-13, injizierte-Frequenz-Recovery). Korrekte Konsequenz: §5 ersetzen + CI-Check, der Doku-Behauptungen gegen vorhandene Artefakte prueft. **[VORBEREITEN: §5 vor Einreichung wirklich korrigieren - das ist eine To-do, keine Redensart.]**

### Woher weiss ICH, dass keine Zahl halluziniert ist? (Verfahren)
"Der Agent prueft sich selbst" ist kein Beweis. Harter Anker = **Reproduzierbarkeit, die ich selbst ausloese:** jede Zahl traegt eine Run-ID nach runs/; ein Befehl regeneriert den Lauf.
Konkret pruefbar: (1) Determinismus (bit-identische Logit-Summe); (2) test_core.py 11/11 inkl. Orthonormalitaet + injizierte-Frequenz-Recovery (strukturell Fabrikations-ausschliessend); (3) Labels-Check. Mein eigenes Verfahren: Lauf neu starten, meta.json gegen UI-Zahl halten, Transition-Schritte aus roher acc-Kurve ablesen statt der gedrafteten Zusammenfassung glauben.
**Grenze:** NICHT jede der 12.769 Attention-Auslesungen einzeln verifiziert - Stichproben + strukturelle Invarianten. Vollstaendige manuelle Verifikation jeder Zahl ist nicht erfolgt und behaupte ich nicht.

### Verstehe ich die Methode oder ist sie eine Blackbox?
**Konzept und Interpretation eigenstaendig, exakte Implementierung delegiert** - das ist die ehrliche Trennlinie. Mit frischem Embedding-Tensor: Embeddingmatrix (p×d_model) in orthonormale Fourier-Basis ueber Token-Achse projizieren, pro Frequenz k Power summieren, normieren, top-8-Anteil berechnen. Entscheidungsregel: ~76% auf top-8 (vs ~32%) = sparse periodische Trig-Signatur = gegrokt; breite Verteilung = nur memoriert. Bestaetigung via PCA (gegrokt = periodischer Ring). Mathematik (Orthonormalitaet, Parseval) verstanden; exakte API-Syntax teils Agent-abhaengig - das sage ich ehrlich.

### Was war fuer DICH persoenlich am schwersten?
Nicht Code schreiben, sondern **Misstrauen-Management gegen plausibel aussehende, aber falsche Ergebnisse** - der Hauptfehlermodus, den PROMPT.md §4 als groesstes Risiko benennt ("einem ML-Ergebnis glauben, das nicht da ist"). Konkret: Der Agent konnte jederzeit eine "gegrokte" Kurve oder Fourier-"Ringe" produzieren, die oberflaechlich ueberzeugen. Meine nicht-delegierbare Aufgabe: entscheiden, wann ein Ergebnis als echt gilt (zwei Schrittindizes 145 vs 8367? Struktur ueber >=3 Seeds robust? Determinismus bit-identisch?). Konkretes Hindernis: die Grokfast-vs-un-beschleunigt-Verwechslung in der Doku erkennen und die Referenz richtig setzen. **Ehrlich:** kein stundenlanges Segfault-Debugging - der schwere Teil war intellektuell-disziplinarisch, nicht handwerklich. Ob das als "Schwierigkeitsgrad" genuegt, ist eine faire Wertungsfrage; ich verkaufe es nicht groesser.

### BWKI-Eigenleistungsschwelle: Verteidigung + Risiko
**Verteidigung (zwei Saetze):** Mein Eigenanteil ist klar abgegrenzt und dokumentiert - Problemdefinition + wissenschaftliche Constitution, Experimentauswahl (A statt Fallback B), Faktentreue-Verifikation gegen geloggte Kurven, plus durchgaengige KI-Offenlegung mit Zeile-fuer-Zeile-Tabelle (= geforderte Nachvollziehbarkeit). **Ehrliches Risiko:** Eine strenge Jury kann argumentieren, "Direktion + Review" liege unterhalb der Schwelle, weil ich weder substantiellen Code selbst geschrieben noch eine eigene Methode entwickelt habe. Mein staerkstes Gegenargument: die wissenschaftliche Urteilskraft (was gilt als echtes Grokking, wann ist ein Ergebnis robust, welche Zahl ist die Referenz) ist die eigentlich pruefbare Leistung - aber das ist eine Wertungsfrage, die gegen mich ausgehen kann, und ich verschleiere das nicht.

### Was wuerde ich aendern, um meinen Anteil groesser/nachweisbarer zu machen?
1. Fehlenden **Ablations-Sweep** (wd-Reihe + mit/ohne LayerNorm) selbst designen und fahren - Kausalitaet in DIESEM Repo statt nur Literatur.
2. Cross-Arch von einem Punkt zu einem echten **Sweep** ueber wd und train_frac; MLP-Konfundierung durch fairere Variante (additive Eingabe) kontrollieren.
3. **Blinder Progress-Measure-Test:** Schluesselfrequenzen NICHT aus demselben Embedding-Spektrum waehlen -> held-out-Unabhaengigkeit sauberer zeigen.
4. **Entscheidungs-Logbuch** mit Zeitstempel pro wissenschaftlicher Weichenstellung - macht meinen Anteil pruefbar statt nur behauptbar.

---

## Block 8: Praktische Relevanz & Wissenschaftskommunikation

### "KI-Sicherheit/Interpretierbarkeit" - Marketing? (KILLER #8)
**Berechtigt, will ich nicht ueberhoehen. Direkt rettet mein Projekt keine reale KI.** Der Sicherheitsbezug ist **konzeptionell, nicht praktisch:** Interpretierbarkeit will verstehen, WAS ein Modell intern berechnet, statt es als Blackbox zu behandeln. Grokking ist das saubere Labor, weil man die EINE gelernte Schaltung vollstaendig aufdecken kann. Zwei uebertragbare Lektionen: (1) Gute Testwerte allein sagen nicht, ob memoriert oder verstanden wurde. (2) Ich habe from-inside-Fortschrittsmasse (restricted/excluded) implementiert, die OHNE separaten Testdatensatz erkennen, ob sich echte Struktur bildet - genau das Ziel grosser Interpretierbarkeit. **Grenze:** Ob das auf Milliarden-Parameter-Modelle skaliert, zeigt mein Projekt nicht - offene Forschung.

### Ist eine Reproduktion + Visualisierung preiswuerdig?
Nackte Reproduktion allein waere keine Auszeichnung wert. Preiswuerdig macht es das Zusammenspiel:
1. **Ehrliche, kanonische Reproduktion** (un-beschleunigt, Nanda-Rezept) mit Determinismus-Nachweis + 11/11 Checks - wissenschaftliches Handwerk, das vielen Schuelerprojekten fehlt.
2. **Neu implementierte interne Fortschrittsmasse** (restricted/excluded, recon ~1e-13).
3. **Eigenes Experiment mit echtem, seed-robustem Befund** (Transformer ~2.9x schneller, sparser).
4. **Visualisierung als Lesehilfe, nicht Schmuck** - macht abstrakte Mathematik fuer Laien begreifbar.
Reproduktion ist hier das Fundament fuer Vertrauen, nicht das Endprodukt.

### "Bunte 3D-Grafik beweist nichts"
Hinter jeder Visualisierung steht eine ruecktracebare Zahl. Drei Belege gegen Schoenfaerberei: (1) Determinismus (bit-identische Logit-Summe - Grafik nicht lauf-zu-lauf willkuerlich); (2) 11/11 Sanity-Checks inkl. injizierter-Frequenz-Recovery + recon ~1e-13 (luegende Fourier-Analyse fiele durch); (3) ich berichte auch unschoene Ergebnisse (Test-Loss steigt im Plateau auf ~26 vor dem Fall auf 0.06). Visualisierung ist Lesehilfe auf gepruefte Zahlen, kein Ersatz.

### Plateau: woher weiss ich, dass innen etwas passiert? (Laien-Bild)
Bild: Ein Schueler kritzelt waehrend einer langen Pause unsichtbar an einer Loesungsmethode. Von aussen (Noten auf neue Aufgaben) sieht man nichts - Zufallsniveau. Aber ich kann ins Heft schauen: Ein Mass misst nur die Bausteine der richtigen Methode (Schluesselfrequenzen), das andere alles Uebrige. Im Plateau faellt das erste Richtung echter Leistung, das zweite steigt stark - das Modell baut im Verborgenen die richtige Schaltung auf und raeumt Auswendiglern-Ballast weg, lange bevor die Testnote springt. Diese Masse brauchen keinen separaten Testdatensatz. Kern: **Stillstand aussen, sichtbarer Fortschritt innen.**

### Browser-LiveLab - was verifiziert es, ist 0.912 "gegrokt"?
Verifiziert eine **bescheidene Behauptung:** das Grokking-PHAENOMEN (train frueh hoch, test lange Zufall, dann Sprung) tritt auf einem Spielzeug-Problem ((a+b) mod 23) live im Browser auf (TensorFlow.js), per Slider (weight decay, train fraction) selbst ausloesbar - keine Aufzeichnung, Training laeuft pro Sitzung neu. Belegt durch automatisierten Headless-Test (verifiziert grokt, test acc **0.912**). 0.912 ist deutlich ueber Zufall (1/23 ~ 0.043) und ueber Memorisierungs-Baseline -> echte Generalisierung, aber NICHT die ~0.98+ der CPU-Reproduktion. **Didaktisches Reproduktions-Artefakt, kein wissenschaftliches Hauptresultat** - die wissenschaftlichen Zahlen kommen ausschliesslich aus den geseedeten PyTorch-Laeufen. p=23, float32, Browser-RNG = Preis fuer Interaktivitaet; ich uebertrage von dort keine quantitativen Aussagen.

### Der EINE Satz Erkenntnisgewinn (nicht bei Power/Nanda)
Mein eigener, quantitativer Vergleichsbefund: Unter identischen, fairen Bedingungen grokt ein 1-Schicht-Transformer ~2.9x schneller (750 vs 2163 Schritte) und mit messbar sparserer, periodischerer interner Darstellung (top-8-Anteil 0.59 vs 0.35) als ein 2-Schicht-MLP, und dieser Unterschied ist groesser als die Seed-Streuung. Einschraenkung im selben Atemzug: ein einzelner Konfigurationspunkt, kein Sweep, teils architektonisch mitbedingt. Aber als reproduzierbare, geseedete Beobachtung ein eigener Datenpunkt, den Power/Nanda so nicht liefern.

---

## Block 9: Gesamtbilanz - was traegt, was nicht (fuer die Schlussfrage)

### Drei Sicherheitsstufen
- **Traegt sicher:** (1) Grokking EXISTIERT in meinem Setup, deterministisch reproduzierbar (kein statistischer Anspruch, bit-exaktes Faktum, 11/11 Checks). (2) Qualitative mechanistische Signatur (sparse Fourier-Konzentration, Ring-Geometrie, 50/50-Attention ueber alle 12.769 Eingaben) robust + ueber Seeds wiederkehrend in der STRUKTUR (Frequenzen variieren, Sparsity nicht).
- **Dazwischen:** Das 2.9x-Architekturverhaeltnis - seed-stabil in Richtung + Groessenordnung (n=3), aber an einem Grokfast-Punkt, nicht auf die un-beschleunigte Welt verallgemeinerbar.
- **Nur Einzelbeobachtung (so etikettiert):** Die exakte Headline-Luecke 8222 Schritte = n=1 (un-beschleunigt, frac=0.3). Groessenordnung solide, genaue Zahl ein Einzellauf.

### Ehrliches Argument GEGEN einen Preis (und warum trotzdem)
**Gegen:** Der wissenschaftliche Kern (Grokking, Kreis-/Fourier-Erklaerung) stammt aus existierender Literatur, umgesetzt mit KI-Unterstuetzung. Wer Originalitaet eng als voellig neue Entdeckung definiert, kann dagegen sein.
**Trotzdem dafuer:** (1) seltener Rigor-Grad (Determinismus, 11/11, jede Zahl ruecktracebar, negative Ergebnisse berichtet); (2) echter eigener Befund (Transformer vs MLP, Mehrfach-Seeds); (3) neu implementierte interne Fortschrittsmasse; (4) didaktischer Beitrag (schwieriges Interpretierbarkeits-Thema fuer Laien + im Browser erfahrbar). Ich beanspruche nicht, Grokking erfunden zu haben - ich beanspruche, es ehrlich, ueberpruefbar und verstaendlich gemacht zu haben.

### Was das Projekt NICHT traegt (nie behaupten)
Eine generalisierte Skalierungs- oder Kausalaussage ueber Architekturen oder ueber die Ursache von Grokking. Diese Grenzen stehen explizit in der Selbstkritik - kaschiere sie nicht.

---

## Anhang: To-do vor Einreichung (markierte Vorbereitungspunkte)
- [ ] RESULTS.md §5 korrigieren (veraltete "nicht implementiert"-Notiz streichen/ersetzen).
- [ ] Changelog-Referenzzahl auf un-beschleunigten Gap 8222 setzen (Grokfast-Lauf-Verwechslung bereinigen).
- [ ] exakten restr_final - full_final-Wert aus JSON parat haben.
- [ ] excluded-loss-Vorlauf in Schritten gegen Schritt 8367 extrahieren (Headline-Upgrade, sonst Grenze offen sagen).
- [ ] k-Sweep {5,6,8,10,12}, Ablations-Sweep (wd, mit/ohne LN), un-beschleunigter MLP-Lauf, Kontroll-MLP (additiv), entkoppelte Seeds, Attention-Zeitreihe: alle als saubere Nachfolge-Schritte benennen koennen (nicht als erledigt behaupten).