# Aggregated analysis tables

Generated 2026-09-06T00:38:55.623363+00:00 at commit `f8fc430e6814c4a38779c1f8371fe28c9e85f4fa` over 20 runs matching `['txf_add_p113_wd1.0_frac0.3_seed?_arch25k', 'mlp_add_p113_wd1.0_frac0.3_seed?_arch25k']`.

Every row is one run at one measurement point. **All seeds are shown** — no row is averaged away (master prompt §16). Numbers are copied from the per-run files named in `source_file`; nothing here is recomputed.

### key_frequencies  (40 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | key_frequencies_used | nanda__n_keys | neuron_clusters__n_keys | embedding_threshold__n_keys |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | — | 56 | 15 | 41 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | — | 13 | 10 | 10 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | — | 56 | 12 | 41 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | — | 11 | 9 | 8 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | — | 54 | 15 | 40 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | — | 11 | 8 | 8 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step008150 | 8150 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | — | 55 | 10 | 42 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | — | 10 | 8 | 8 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | — | 56 | 10 | 42 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | — | 14 | 9 | 8 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step009375 | 9375 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | — | 56 | 14 | 41 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | — | 12 | 10 | 10 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step009900 | 9900 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | — | 56 | 15 | 41 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | — | 12 | 11 | 10 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step009200 | 9200 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | — | 56 | 11 | 42 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | — | 13 | 10 | 10 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step009800 | 9800 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | — | 56 | 10 | 41 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | — | 13 | 9 | 9 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step009300 | 9300 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | — | 56 | 11 | 42 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | — | 9 | 9 | 9 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step007975 | 7975 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | — | 6 | 5 | 32 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | — | 8 | 4 | 4 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step006250 | 6250 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | — | 5 | 5 | 37 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | — | 4 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | — | 6 | 4 | 28 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | — | 6 | 4 | 4 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step007425 | 7425 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | — | 6 | 4 | 30 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | — | 5 | 3 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step010275 | 10275 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | — | 6 | 5 | 33 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | — | 5 | 5 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step005625 | 5625 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | — | 4 | 3 | 36 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | — | 3 | 3 | 3 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step006325 | 6325 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | — | 7 | 5 | 32 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | — | 4 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step007750 | 7750 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | — | 5 | 5 | 35 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | — | 4 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | — | 4 | 4 | 34 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | — | 4 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step005825 | 5825 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | — | 7 | 6 | 33 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | — | 5 | 5 | 5 |

### mlp_mechanism  (20 rows, 0 missing, 10 not applicable to this architecture)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | n_neurons | n_live | structured_n | structured_fraction_of_live |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 512 | 512 | 153 | 0.2988 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 512 | 512 | 451 | 0.8809 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 512 | 512 | 136 | 0.2656 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 512 | 512 | 445 | 0.8691 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 512 | 512 | 147 | 0.2871 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 512 | 512 | 444 | 0.8672 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step008150 | 8150 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 512 | 512 | 118 | 0.2305 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 512 | 512 | 503 | 0.9824 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 512 | 512 | 143 | 0.2793 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 512 | 512 | 479 | 0.9355 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step009375 | 9375 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 512 | 512 | 161 | 0.3145 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 512 | 512 | 451 | 0.8809 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step009900 | 9900 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 512 | 512 | 149 | 0.291 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 512 | 512 | 448 | 0.875 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step009200 | 9200 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 512 | 512 | 142 | 0.2773 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 512 | 512 | 455 | 0.8887 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step009800 | 9800 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 512 | 512 | 139 | 0.2715 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 512 | 512 | 470 | 0.918 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step009300 | 9300 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 512 | 512 | 135 | 0.2637 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 512 | 512 | 465 | 0.9082 |

### transformer_mechanism  (20 rows, 0 missing, 10 not applicable to this architecture)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | n_neurons | n_live | structured_n | structured_fraction_of_live |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step007975 | 7975 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 512 | 512 | 316 | 0.6172 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 512 | 512 | 494 | 0.9648 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step006250 | 6250 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 512 | 512 | 279 | 0.5449 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 512 | 512 | 507 | 0.9902 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 512 | 512 | 309 | 0.6035 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 512 | 512 | 495 | 0.9668 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step007425 | 7425 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 512 | 512 | 320 | 0.625 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 512 | 512 | 509 | 0.9941 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step010275 | 10275 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 512 | 512 | 293 | 0.5723 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 512 | 512 | 502 | 0.9805 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step005625 | 5625 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 512 | 512 | 236 | 0.4609 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 512 | 512 | 512 | 1 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step006325 | 6325 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 512 | 512 | 294 | 0.5742 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 512 | 512 | 505 | 0.9863 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step007750 | 7750 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 512 | 512 | 287 | 0.5605 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 512 | 512 | 499 | 0.9746 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 512 | 512 | 299 | 0.584 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 512 | 512 | 503 | 0.9824 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step005825 | 5825 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 512 | 512 | 329 | 0.6426 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 512 | 512 | 503 | 0.9824 |

### wave_fitting  (20 rows, 0 missing, 10 not applicable to this architecture)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | key_frequencies_used | u_a__fraction_best_aic_sinusoid | u_a__fraction_best_aic_square | u_a__fraction_best_aic_odd_harmonics |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | — | 0.3516 | 0.375 | 0.2441 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | — | 0.3398 | 0.02734 | 0.627 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | — | 0.3613 | 0.3906 | 0.2129 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | — | 0.2539 | 0.01562 | 0.7227 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | — | 0.3652 | 0.3691 | 0.2441 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | — | 0.3184 | 0.03711 | 0.6387 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step008150 | 8150 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | — | 0.3477 | 0.4258 | 0.2031 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | — | 0.4648 | 0.02148 | 0.5137 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | — | 0.3496 | 0.4219 | 0.207 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | — | 0.3379 | 0.01953 | 0.6348 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step009375 | 9375 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | — | 0.3223 | 0.4102 | 0.25 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | — | 0.3379 | 0.02148 | 0.6348 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step009900 | 9900 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | — | 0.3242 | 0.3945 | 0.252 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | — | 0.2441 | 0.01562 | 0.7402 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step009200 | 9200 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | — | 0.3457 | 0.4062 | 0.2324 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | — | 0.2324 | 0.02539 | 0.7383 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step009800 | 9800 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | — | 0.3281 | 0.4141 | 0.2363 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | — | 0.2812 | 0.03125 | 0.6816 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step009300 | 9300 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | — | 0.3301 | 0.4121 | 0.2461 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | — | 0.1758 | 0.009766 | 0.8145 |

### logit_formula_fit  (40 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | family_size | control_set_size | control_n | sparse_sinusoid__r2_test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 56 | 56 | 50 | 0.4994 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 42 | 42 | 50 | 0.9667 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 56 | 56 | 50 | 0.5398 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 35 | 35 | 50 | 0.973 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 56 | 56 | 50 | 0.5104 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 32 | 32 | 50 | 0.9769 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step008150 | 8150 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 56 | 56 | 50 | 0.6375 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 31 | 31 | 50 | 0.9773 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 56 | 56 | 50 | 0.6067 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 37 | 37 | 50 | 0.9772 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step009375 | 9375 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 56 | 56 | 50 | 0.5096 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 38 | 38 | 50 | 0.9686 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step009900 | 9900 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 56 | 56 | 50 | 0.4964 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 34 | 34 | 50 | 0.9619 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step009200 | 9200 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 56 | 56 | 50 | 0.5702 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 34 | 34 | 50 | 0.9719 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step009800 | 9800 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 56 | 56 | 50 | 0.5763 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 36 | 36 | 50 | 0.9743 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step009300 | 9300 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 56 | 56 | 50 | 0.5764 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 27 | 27 | 50 | 0.9752 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step007975 | 7975 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 21 | 21 | 50 | 0.69 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 28 | 28 | 50 | 0.4271 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step006250 | 6250 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 17 | 17 | 50 | 0.7183 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 15 | 15 | 50 | 0.695 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 19 | 19 | 50 | 0.7433 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 18 | 18 | 50 | 0.5016 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step007425 | 7425 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 21 | 21 | 50 | 0.6361 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 18 | 18 | 50 | 0.4786 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step010275 | 10275 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 22 | 22 | 50 | 0.7164 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 18 | 18 | 50 | 0.6116 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step005625 | 5625 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 16 | 16 | 50 | 0.7397 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 12 | 12 | 50 | 0.5637 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step006325 | 6325 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 22 | 22 | 50 | 0.6909 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 14 | 14 | 50 | 0.7311 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step007750 | 7750 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 17 | 17 | 50 | 0.7164 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 14 | 14 | 50 | 0.6943 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 13 | 13 | 50 | 0.7241 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 13 | 13 | 50 | 0.5284 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step005825 | 5825 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 25 | 25 | 50 | 0.6831 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 17 | 17 | 50 | 0.5092 |

### h3_validity  (40 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | h3a__u_a__waveform_sensitivity_top1 | h3a__u_a__waveform_sensitivity_top4 | h3a__u_a__waveform_sensitivity_top8 | h3a__u_a__shape_separation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step008150 | 8150 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step009375 | 9375 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step009900 | 9900 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step009200 | 9200 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step009800 | 9800 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step009300 | 9300 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step007975 | 7975 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step006250 | 6250 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step007425 | 7425 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step010275 | 10275 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step005625 | 5625 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step006325 | 6325 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step007750 | 7750 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step005825 | 5825 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |

### causal_ablation  (40 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | baseline_test_acc | n_structured | g4_passed | g4_remove_structured_necessary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 0.9513 | 153 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 1 | 451 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 0.9508 | 136 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 1 | 445 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 0.9512 | 147 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 1 | 444 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step008150 | 8150 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 0.9541 | 118 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 1 | 503 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 0.9512 | 143 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 1 | 479 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step009375 | 9375 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 0.9507 | 161 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 1 | 451 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step009900 | 9900 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 0.9503 | 149 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 1 | 448 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step009200 | 9200 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 0.9537 | 142 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 1 | 455 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step009800 | 9800 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 0.9509 | 139 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 1 | 470 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step009300 | 9300 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 0.9508 | 135 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 1 | 465 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step007975 | 7975 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 0.9503 | 316 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 0.9977 | 494 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step006250 | 6250 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 0.9505 | 279 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 0.9994 | 507 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.9549 | 309 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.9971 | 495 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step007425 | 7425 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 0.9502 | 320 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 0.9996 | 509 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step010275 | 10275 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 0.9523 | 293 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 1 | 502 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step005625 | 5625 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 0.9526 | 236 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 1 | 512 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step006325 | 6325 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 0.9531 | 294 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 1 | 505 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step007750 | 7750 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 0.9516 | 287 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 1 | 499 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.954 | 299 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.9993 | 503 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step005825 | 5825 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 0.9512 | 329 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 0.9998 | 503 | no | no |

### structure_over_time  (20 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | role__init | role__pre_memorization | role__memorization | role__mid_plateau |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 0 | 0 | 160 | 4000 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 0 | 0 | 160 | 4000 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 0 | 0 | 160 | 5000 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 0 | 0 | 140 | 3000 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 0 | 0 | 150 | 5000 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 0 | 0 | 140 | 3000 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 0 | 0 | 140 | 3000 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 0 | 0 | 140 | 3000 |

### progress_measures  (20 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | progress_measures_step | full_loss_all | legacy_broad_mask__n_kept | legacy_broad_mask__n_removed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 25000 | 7.235e-06 | 729 | 5200 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 25000 | 6.146e-06 | 529 | 4488 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 25000 | 5.336e-06 | 529 | 4488 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 25000 | 4.609e-06 | 441 | 4120 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 25000 | 7.087e-06 | 841 | 5544 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 25000 | 6.437e-06 | 625 | 4848 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 25000 | 7.955e-06 | 625 | 4848 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 25000 | 6.553e-06 | 729 | 5200 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 25000 | 5.123e-06 | 729 | 5200 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 25000 | 5.222e-06 | 361 | 3744 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 25000 | 0.007261 | 289 | 3360 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 25000 | 0.002142 | 81 | 1744 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 25000 | 0.01162 | 169 | 2568 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 25000 | 0.003213 | 121 | 2160 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 25000 | 1.779e-05 | 121 | 2160 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 25000 | 0.002729 | 49 | 1320 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 25000 | 3.154e-05 | 81 | 1744 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 25000 | 0.0008746 | 81 | 1744 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 25000 | 0.0015 | 81 | 1744 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 25000 | 0.0006053 | 121 | 2160 |

### bounded_alternative  (20 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | key_frequencies_used | n_hidden_units | probe_linear__test_acc | probe_linear__control_test_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | — | 512 | 1 | 0.008727 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | — | 512 | 0.9999 | 0.0122 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | — | 512 | 0.9998 | 0.00951 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | — | 512 | 0.9999 | 0.009734 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | — | 512 | 1 | 0.008279 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | — | 512 | 0.9999 | 0.008727 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | — | 512 | 0.9998 | 0.009398 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | — | 512 | 1 | 0.008279 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | — | 512 | 1 | 0.006825 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | — | 512 | 1 | 0.009846 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | — | 512 | 0.9999 | 0.00772 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | — | 512 | 1 | 0.00772 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | — | 512 | 1 | 0.008503 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | — | 512 | 1 | 0.007272 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | — | 512 | 1 | 0.00951 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | — | 512 | 1 | 0.009398 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | — | 512 | 1 | 0.007496 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | — | 512 | 0.9996 | 0.007608 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | — | 512 | 1 | 0.01007 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | — | 512 | 1 | 0.008055 |
