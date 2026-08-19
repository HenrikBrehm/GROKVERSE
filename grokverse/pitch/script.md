# GROKVERSE — Spoken Pitch Script

**Total runtime: ~3:30–4:30 min.** Timings are approximate and marked per section.
All numbers below are verified against `runs/txf_add_p113_wd1.0_frac0.3_seed0/run.json`
(the canonical **un-accelerated** reproduction — the explorer's default run) and
`RESULTS.md`; the cross-architecture numbers come from the matched Grokfast runs
cited in `RESULTS.md` §3.

---

## 0. HOOK — [0:00–0:08]

> "Watch the exact moment an AI 'gets it'."

*(Beat. Let the line land. Screen is black, then fades to the explorer.)*

---

## 1. WHAT IS GROKKING? — [0:08–0:50] (~40s)

When you train a neural network, you normally expect it to slowly get better at its task — the more it practices, the more it generalizes. But sometimes something strange happens. The network first *memorizes* its training examples perfectly, scoring 100% on what it has seen — and then, on data it has never seen, it stays completely lost. For a long time. And then, abruptly, long after you'd have given up, it suddenly *understands*. Test performance snaps from near-random to near-perfect. The model stopped memorizing and discovered the underlying rule. Researchers call this **grokking** — and that delayed, almost magical click is exactly what we set out to capture, reproduce, and *show you*.

---

## 2. OUR HONEST REPRODUCTION — [0:50–1:45] (~55s)

We trained a tiny 1-layer transformer on a clean toy problem: modular addition, mod 113. Tiny enough to understand completely; rich enough to grok. No tricks, no acceleration — the faithful setup from the literature, reproduced on an ordinary CPU in about 24 minutes.

And we caught it in the act. Here are the exact numbers from our run:

- By **step 145**, the model has **memorized** — training accuracy crosses 99%, and is perfect by step 156. On unseen test data it's still essentially guessing.
- Then the plateau. For more than **eight thousand steps**, nothing seems to happen — test loss even climbs.
- At **step 8,367**, test accuracy crosses **0.95** — generalization arrives, suddenly.
- That delay between memorizing and understanding — the **grok gap** — is **8,222 steps**, measured, not estimated. Final test accuracy: **0.981**, with test loss collapsing from a plateau peak near **26** down to **0.06**.

These aren't numbers we wished into existence. They're logged, seeded, and reproducible: when we deterministically re-trained this exact run for the mechanistic analysis, it recovered the very same transition — memorize at 145, generalize at 8,367, to the step. And the same phenomenon repeats across seeds — accelerated corroboration runs grok every single time.

---

## 3. THE MECHANISTIC STORY — [1:45–2:40] (~55s)

But the real question isn't *when* it groks — it's *what changes inside*.

When you look at the token embeddings — how the network represents the numbers 0 through 112 — something beautiful happens. During memorization, they're a disorganized blob. As the model groks, the embeddings **reorganize into a periodic structure**. In the Fourier domain, the representation concentrates onto a handful of key frequencies: in our canonical run, the top eight hold **76%** of the total frequency power — against just 32% for a matched run that never grokked. And the attention pattern shows the read-out position pulling in **both operands, almost exactly 50/50** — the wiring of a circuit that must combine a and b.

We went one level deeper and reproduced the field's sharpest test, the **restricted and excluded loss**: keep *only* those eight key frequencies, and the network still solves the task almost perfectly. Delete *just* those eight, and it collapses to far worse than random guessing. That's not decoration — that *is* the algorithm. The network rediscovered that modular addition is naturally expressed with rotations and waves, the same trick a mathematician would use. And we can show you that structure forming, live, in 3D.

---

## 4. THE INTERACTIVE EXPLORER — [2:40–3:20] (~40s)

So we built an explorer where you don't just read about grokking — you *scrub* it.

Drag the timeline. On the left, the embedding cloud: at the start it's a shapeless **blob**. As you scrub past the grok point, those same token points **pull themselves into a ring** — the periodic structure, made visible.

Next to it, the loss curves, with a gold marker dropped exactly at **step 8,367**, the moment generalization landed — and a dedicated panel where you can watch the restricted and excluded losses split apart as the circuit forms, live against the scrubber. Switch runs — including an honestly-labeled run that *never* groks, so you can see the difference — or open the **Live Lab** and train a network in your own browser: raise the weight decay and grokking appears, with the delay measured on screen.

---

## 5. HONEST SELF-CRITIQUE — [3:20–4:00] (~40s)

Now — we want to be rigorous, so here's what we'd flag to a referee.

First: our original cross-architecture experiment. Under matched accelerated settings, a transformer groks this task about **three times faster** than an MLP. When we re-ran the whole comparison in the honest, un-accelerated setting, the transformer still won in **every single seed** — and still landed on a visibly **sparser** frequency circuit — but the speed gap shrank to about **1.3×**. We report both: the sparsity difference is the robust finding; the headline speed ratio was partly a property of the accelerated setting.

Second: bit-exact determinism is real but *environment-scoped* — change the CPU thread count and the last decimal places drift, which is why every run records its thread count, and why we lean on the transition steps, which reproduce exactly.

That honesty is the point. This is an interpretability project — if we fudged the numbers, we'd be defeating our own thesis.

---

## 6. CLOSE — [4:00–4:15] (~15s)

Grokking is one of the clearest windows we have into *how* a network learns a real algorithm instead of memorizing. GROKVERSE makes that window something you can see, scrub, and trust.

> "Watch the exact moment an AI gets it." Now you can.

*(Hold on the ring. Fade.)*
