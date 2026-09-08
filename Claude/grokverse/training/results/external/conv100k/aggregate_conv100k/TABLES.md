# Aggregated analysis tables

Generated 2026-09-05T20:49:52.892771+00:00 at commit `42dd79a49225abe33f1b4f144acb98ec26d47f69` over 20 runs matching `['txf_add_p113_wd1.0_frac0.3_seed?_conv100k', 'mlp_add_p113_wd1.0_frac0.3_seed?_conv100k']`.

Every row is one run at one measurement point. **All seeds are shown** — no row is averaged away (master prompt §16). Numbers are copied from the per-run files named in `source_file`; nothing here is recomputed.

### key_frequencies  (40 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | key_frequencies_used | nanda__n_keys | neuron_clusters__n_keys | embedding_threshold__n_keys |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step009250 | 9250 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | — | 56 | 17 | 41 |
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | — | 11 | 8 | 8 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step009775 | 9775 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | — | 56 | 12 | 41 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | — | 11 | 9 | 9 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step009000 | 9000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | — | 55 | 14 | 40 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | — | 11 | 8 | 7 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | — | 55 | 10 | 42 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | — | 9 | 8 | 7 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step008675 | 8675 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | — | 56 | 12 | 42 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | — | 9 | 7 | 7 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | — | 56 | 14 | 41 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | — | 12 | 8 | 7 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step009725 | 9725 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | — | 55 | 13 | 40 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | — | 12 | 9 | 8 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step009325 | 9325 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | — | 55 | 13 | 41 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | — | 11 | 9 | 8 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step009700 | 9700 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | — | 56 | 11 | 41 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | — | 10 | 8 | 7 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step009550 | 9550 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | — | 56 | 12 | 41 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | — | 12 | 8 | 7 |
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | step008100 | 8100 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | — | 5 | 4 | 33 |
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | — | 5 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | step006650 | 6650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | — | 5 | 4 | 36 |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | — | 4 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | — | 6 | 6 | 24 |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | — | 5 | 5 | 6 |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | step007650 | 7650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | — | 7 | 5 | 29 |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | — | 5 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | step010150 | 10150 | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | — | 7 | 4 | 34 |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | — | 4 | 4 | 4 |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | step006275 | 6275 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | — | 3 | 3 | 35 |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | — | 3 | 3 | 3 |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | step005425 | 5425 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | — | 6 | 5 | 33 |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | — | 5 | 5 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | step008100 | 8100 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | — | 4 | 4 | 36 |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | — | 4 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | step008175 | 8175 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | — | 4 | 4 | 34 |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | — | 4 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | step005750 | 5750 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | — | 6 | 5 | 34 |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | — | 5 | 5 | 5 |

### mlp_mechanism  (20 rows, 0 missing, 10 not applicable to this architecture)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | n_neurons | n_live | structured_n | structured_fraction_of_live |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step009250 | 9250 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | 512 | 512 | 157 | 0.3066 |
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | 512 | 512 | 512 | 1 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step009775 | 9775 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | 512 | 512 | 140 | 0.2734 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | 512 | 512 | 509 | 0.9941 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step009000 | 9000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | 512 | 512 | 139 | 0.2715 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | 512 | 512 | 468 | 0.9141 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | 512 | 512 | 132 | 0.2578 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | 512 | 512 | 508 | 0.9922 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step008675 | 8675 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | 512 | 512 | 136 | 0.2656 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | 512 | 512 | 512 | 1 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | 512 | 512 | 146 | 0.2852 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | 512 | 512 | 507 | 0.9902 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step009725 | 9725 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | 512 | 512 | 146 | 0.2852 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | 512 | 512 | 512 | 1 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step009325 | 9325 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | 512 | 512 | 137 | 0.2676 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | 512 | 512 | 512 | 1 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step009700 | 9700 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | 512 | 512 | 130 | 0.2539 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | 512 | 512 | 512 | 1 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step009550 | 9550 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | 512 | 512 | 128 | 0.25 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | 512 | 512 | 512 | 1 |

### transformer_mechanism  (20 rows, 0 missing, 10 not applicable to this architecture)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | n_neurons | n_live | structured_n | structured_fraction_of_live |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | step008100 | 8100 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 512 | 512 | 275 | 0.5371 |
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 512 | 512 | 512 | 1 |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | step006650 | 6650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | 512 | 512 | 287 | 0.5605 |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | 512 | 512 | 512 | 1 |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | 512 | 512 | 325 | 0.6348 |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | 512 | 512 | 412 | 0.8047 |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | step007650 | 7650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | 512 | 512 | 291 | 0.5684 |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | 512 | 512 | 503 | 0.9824 |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | step010150 | 10150 | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | 512 | 512 | 277 | 0.541 |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | 512 | 512 | 512 | 1 |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | step006275 | 6275 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | 512 | 512 | 240 | 0.4688 |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | 512 | 512 | 512 | 1 |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | step005425 | 5425 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | 512 | 512 | 312 | 0.6094 |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | 512 | 512 | 511 | 0.998 |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | step008100 | 8100 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 512 | 512 | 297 | 0.5801 |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 512 | 512 | 512 | 1 |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | step008175 | 8175 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | 512 | 512 | 280 | 0.5469 |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | 512 | 512 | 511 | 0.998 |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | step005750 | 5750 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | 512 | 512 | 286 | 0.5586 |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | 512 | 512 | 512 | 1 |

### wave_fitting  (20 rows, 0 missing, 10 not applicable to this architecture)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | key_frequencies_used | u_a__fraction_best_aic_sinusoid | u_a__fraction_best_aic_square | u_a__fraction_best_aic_odd_harmonics |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step009250 | 9250 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | — | 0.3496 | 0.3633 | 0.2637 |
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | — | 0.2109 | 0.007812 | 0.7754 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step009775 | 9775 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | — | 0.3418 | 0.4082 | 0.2266 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | — | 0.2422 | 0 | 0.7578 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step009000 | 9000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | — | 0.3633 | 0.3926 | 0.2246 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | — | 0.3711 | 0 | 0.6289 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | — | 0.3516 | 0.4297 | 0.1934 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | — | 0.293 | 0 | 0.707 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step008675 | 8675 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | — | 0.3418 | 0.4062 | 0.2207 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | — | 0.2559 | 0 | 0.7441 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | — | 0.3496 | 0.3965 | 0.2266 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | — | 0.2324 | 0.009766 | 0.7129 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step009725 | 9725 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | — | 0.3418 | 0.3652 | 0.2637 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | — | 0.1465 | 0 | 0.8535 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step009325 | 9325 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | — | 0.3301 | 0.4082 | 0.2383 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | — | 0.207 | 0 | 0.793 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step009700 | 9700 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | — | 0.3555 | 0.4082 | 0.209 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | — | 0.2773 | 0 | 0.7227 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step009550 | 9550 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | — | 0.3555 | 0.3965 | 0.2305 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | — | 0.09961 | 0 | 0.9004 |

### logit_formula_fit  (40 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | family_size | control_set_size | control_n | sparse_sinusoid__r2_test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step009250 | 9250 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | 56 | 56 | 50 | 0.4718 |
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | 35 | 35 | 50 | 0.9797 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step009775 | 9775 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | 56 | 56 | 50 | 0.5408 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | 33 | 33 | 50 | 0.9672 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step009000 | 9000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | 56 | 56 | 50 | 0.5394 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | 33 | 33 | 50 | 0.9788 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | 56 | 56 | 50 | 0.6453 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | 30 | 30 | 50 | 0.9719 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step008675 | 8675 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | 56 | 56 | 50 | 0.6111 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | 28 | 28 | 50 | 0.9864 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | 56 | 56 | 50 | 0.5288 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | 39 | 39 | 50 | 0.9802 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step009725 | 9725 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | 56 | 56 | 50 | 0.5129 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | 35 | 35 | 50 | 0.9703 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step009325 | 9325 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | 56 | 56 | 50 | 0.5541 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | 34 | 34 | 50 | 0.9754 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step009700 | 9700 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | 56 | 56 | 50 | 0.6089 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | 30 | 30 | 50 | 0.9801 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step009550 | 9550 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | 56 | 56 | 50 | 0.5519 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | 33 | 33 | 50 | 0.9806 |
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | step008100 | 8100 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 18 | 18 | 50 | 0.7156 |
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 18 | 18 | 50 | 0.6343 |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | step006650 | 6650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | 17 | 17 | 50 | 0.7014 |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | 15 | 15 | 50 | 0.6312 |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | 20 | 20 | 50 | 0.6822 |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | 16 | 16 | 50 | 0.6469 |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | step007650 | 7650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | 23 | 23 | 50 | 0.7067 |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | 17 | 17 | 50 | 0.5725 |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | step010150 | 10150 | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | 24 | 24 | 50 | 0.7266 |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | 14 | 14 | 50 | 0.6485 |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | step006275 | 6275 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | 12 | 12 | 50 | 0.7652 |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | 12 | 12 | 50 | 0.5607 |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | step005425 | 5425 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | 20 | 20 | 50 | 0.666 |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | 18 | 18 | 50 | 0.4656 |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | step008100 | 8100 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 14 | 14 | 50 | 0.7007 |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 14 | 14 | 50 | 0.6922 |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | step008175 | 8175 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | 13 | 13 | 50 | 0.7227 |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | 13 | 13 | 50 | 0.4773 |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | step005750 | 5750 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | 21 | 21 | 50 | 0.7049 |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | 19 | 19 | 50 | 0.5218 |

### h3_validity  (40 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | h3a__u_a__waveform_sensitivity_top1 | h3a__u_a__waveform_sensitivity_top4 | h3a__u_a__waveform_sensitivity_top8 | h3a__u_a__shape_separation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step009250 | 9250 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step009775 | 9775 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step009000 | 9000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step008675 | 8675 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step009725 | 9725 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step009325 | 9325 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step009700 | 9700 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step009550 | 9550 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | step008100 | 8100 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | step006650 | 6650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | step007650 | 7650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | step010150 | 10150 | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | step006275 | 6275 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | step005425 | 5425 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | step008100 | 8100 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | step008175 | 8175 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | step005750 | 5750 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |

### causal_ablation  (40 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | baseline_test_acc | n_structured | g4_passed | g4_remove_structured_necessary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step009250 | 9250 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | 0.9503 | 157 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | 1 | 512 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step009775 | 9775 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | 0.9525 | 140 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | 1 | 509 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step009000 | 9000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | 0.9516 | 139 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | 1 | 468 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | 0.9508 | 132 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | 1 | 508 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step008675 | 8675 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | 0.9509 | 136 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | 1 | 512 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | 0.9517 | 146 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | 1 | 507 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step009725 | 9725 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | 0.9518 | 146 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | 1 | 512 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step009325 | 9325 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | 0.9525 | 137 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | 1 | 512 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step009700 | 9700 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | 0.9516 | 130 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | 1 | 512 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step009550 | 9550 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | 0.9504 | 128 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | 1 | 512 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | step008100 | 8100 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 0.9521 | 275 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 1 | 512 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | step006650 | 6650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | 0.952 | 287 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | 1 | 512 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | 0.9511 | 325 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | 0.9998 | 412 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | step007650 | 7650 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | 0.9502 | 291 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | 1 | 503 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | step010150 | 10150 | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | 0.9505 | 277 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | 1 | 512 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | step006275 | 6275 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | 0.9555 | 240 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | 0.9998 | 512 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | step005425 | 5425 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | 0.9552 | 312 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | 1 | 511 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | step008100 | 8100 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 0.9556 | 297 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 1 | 512 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | step008175 | 8175 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | 0.9555 | 280 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | 0.9992 | 511 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | step005750 | 5750 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | 0.9512 | 286 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | step100000 | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | 0.9998 | 512 | no | no |

### structure_over_time  (20 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | role__init | role__pre_memorization | role__memorization | role__mid_plateau |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | 0 | 0 | 160 | 4000 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | 0 | 0 | 160 | 4000 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | 0 | 0 | 160 | 5000 |
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | 0 | 0 | 140 | 3000 |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | 0 | 0 | 150 | 5000 |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | 0 | 0 | 140 | 3000 |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | 0 | 0 | 140 | 3000 |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | all_checkpoints | 100000 | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | 0 | 0 | 140 | 3000 |

### progress_measures  (20 rows, 0 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | progress_measures_step | full_loss_all | legacy_broad_mask__n_kept | legacy_broad_mask__n_removed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0_conv100k | mlp | 0 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9250 | 100000 | 4.564e-06 | 529 | 4488 |
| mlp_add_p113_wd1.0_frac0.3_seed1_conv100k | mlp | 1 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9775 | 100000 | 5.479e-06 | 529 | 4488 |
| mlp_add_p113_wd1.0_frac0.3_seed2_conv100k | mlp | 2 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9000 | 100000 | 4.804e-06 | 529 | 4488 |
| mlp_add_p113_wd1.0_frac0.3_seed3_conv100k | mlp | 3 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8125 | 100000 | 4.44e-06 | 361 | 3744 |
| mlp_add_p113_wd1.0_frac0.3_seed4_conv100k | mlp | 4 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 8675 | 100000 | 4.5e-06 | 361 | 3744 |
| mlp_add_p113_wd1.0_frac0.3_seed5_conv100k | mlp | 5 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9175 | 100000 | 4.096e-06 | 625 | 4848 |
| mlp_add_p113_wd1.0_frac0.3_seed6_conv100k | mlp | 6 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9725 | 100000 | 4.814e-06 | 625 | 4848 |
| mlp_add_p113_wd1.0_frac0.3_seed7_conv100k | mlp | 7 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9325 | 100000 | 5.144e-06 | 529 | 4488 |
| mlp_add_p113_wd1.0_frac0.3_seed8_conv100k | mlp | 8 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9700 | 100000 | 4.333e-06 | 441 | 4120 |
| mlp_add_p113_wd1.0_frac0.3_seed9_conv100k | mlp | 9 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 160 | 9550 | 100000 | 4.102e-06 | 625 | 4848 |
| txf_add_p113_wd1.0_frac0.3_seed0_conv100k | transformer | 0 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 100000 | 1.246e-05 | 121 | 2160 |
| txf_add_p113_wd1.0_frac0.3_seed1_conv100k | transformer | 1 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6650 | 100000 | 2.358e-05 | 81 | 1744 |
| txf_add_p113_wd1.0_frac0.3_seed2_conv100k | transformer | 2 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8650 | 100000 | 0.0007181 | 121 | 2160 |
| txf_add_p113_wd1.0_frac0.3_seed3_conv100k | transformer | 3 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 7650 | 100000 | 0.0001501 | 121 | 2160 |
| txf_add_p113_wd1.0_frac0.3_seed4_conv100k | transformer | 4 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 150 | 10150 | 100000 | 0.0001359 | 81 | 1744 |
| txf_add_p113_wd1.0_frac0.3_seed5_conv100k | transformer | 5 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 6275 | 100000 | 0.0008959 | 49 | 1320 |
| txf_add_p113_wd1.0_frac0.3_seed6_conv100k | transformer | 6 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5425 | 100000 | 0.0002531 | 121 | 2160 |
| txf_add_p113_wd1.0_frac0.3_seed7_conv100k | transformer | 7 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8100 | 100000 | 2.099e-06 | 81 | 1744 |
| txf_add_p113_wd1.0_frac0.3_seed8_conv100k | transformer | 8 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 8175 | 100000 | 0.002808 | 81 | 1744 |
| txf_add_p113_wd1.0_frac0.3_seed9_conv100k | transformer | 9 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 100000 | 140 | 5750 | 100000 | 0.0007913 | 121 | 2160 |

### bounded_alternative  (0 rows, 20 missing)
_No rows._

**Missing (20):** mlp_add_p113_wd1.0_frac0.3_seed0_conv100k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_seed1_conv100k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_seed2_conv100k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_seed3_conv100k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_seed4_conv100k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_seed5_conv100k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_seed6_conv100k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_seed7_conv100k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_seed8_conv100k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_seed9_conv100k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_seed0_conv100k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_seed1_conv100k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_seed2_conv100k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_seed3_conv100k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_seed4_conv100k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_seed5_conv100k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_seed6_conv100k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_seed7_conv100k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_seed8_conv100k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_seed9_conv100k (no analysis/bounded_alternative/*.json)
