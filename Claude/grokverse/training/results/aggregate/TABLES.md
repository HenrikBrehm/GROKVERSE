# Aggregated analysis tables

Generated 2026-09-04T02:59:15.806983+00:00 at commit `8b312a2db2a070d4fc3a33de5ac898c3f55ddbba` over 51 runs matching `*_arch25k`.

Every row is one run at one measurement point. **All seeds are shown** — no row is averaged away (master prompt §16). Numbers are copied from the per-run files named in `source_file`; nothing here is recomputed.

### key_frequencies  (80 rows, 11 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | key_frequencies_used | nanda__n_keys | neuron_clusters__n_keys | embedding_threshold__n_keys |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | — | 56 | 15 | 43 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step002100 | 2100 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | — | 56 | 17 | 43 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | — | 56 | 15 | 41 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step010850 | 10850 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | — | 56 | 14 | 41 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | — | 24 | 11 | 15 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | — | 13 | 10 | 10 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | — | 11 | 10 | 10 |
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | — | 12 | 9 | 9 |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step001725 | 1725 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | — | 56 | 16 | 43 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step002125 | 2125 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | — | 56 | 17 | 44 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | — | 56 | 12 | 41 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step011475 | 11475 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | — | 56 | 15 | 40 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | — | 17 | 11 | 14 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | — | 11 | 9 | 8 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | — | 9 | 9 | 8 |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | — | 9 | 9 | 9 |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | — | 56 | 16 | 43 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step002175 | 2175 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | — | 56 | 16 | 43 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | — | 54 | 15 | 40 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step010125 | 10125 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | — | 56 | 12 | 41 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | — | 17 | 9 | 13 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | — | 11 | 8 | 8 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | — | 10 | 9 | 8 |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | — | 10 | 9 | 9 |
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
| m2h_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp_twohot | 0 | step012775 | 12775 | 0.3 | 1 | no | 25 | 10 | 25000 | 230 | 12775 | — | 56 | 17 | 48 |
| m2h_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp_twohot | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 230 | 12775 | — | 56 | 14 | 49 |
| m2h_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp_twohot | 1 | step013300 | 13300 | 0.3 | 1 | no | 25 | 10 | 25000 | 240 | 13300 | — | 56 | 17 | 49 |
| m2h_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp_twohot | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 240 | 13300 | — | 56 | 15 | 49 |
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | step000375 | 375 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | — | 3 | 3 | 45 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | step000675 | 675 | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | — | 5 | 5 | 43 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | step006975 | 6975 | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | — | 5 | 4 | 34 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step007975 | 7975 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | — | 6 | 5 | 32 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | — | 4 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | — | 8 | 4 | 4 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | — | 4 | 5 | 5 |
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | — | 3 | 3 | 3 |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | step000675 | 675 | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | — | 7 | 7 | 43 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | step000800 | 800 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | — | 3 | 3 | 41 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step006250 | 6250 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | — | 5 | 5 | 37 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | step006500 | 6500 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | — | 4 | 4 | 35 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | — | 4 | 4 | 4 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | — | 4 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | — | 6 | 3 | 3 |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | — | 4 | 4 | 5 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | step000725 | 725 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | — | 2 | 2 | 41 |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | step000875 | 875 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | — | 7 | 4 | 40 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | step006500 | 6500 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | — | 7 | 6 | 36 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | — | 6 | 4 | 28 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | — | 5 | 5 | 6 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | — | 6 | 4 | 4 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | — | 3 | 2 | 2 |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | — | 4 | 4 | 5 |
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

**Missing (11):** m2h_add_p113_wd1.0_frac0.3_seed2_arch25k (no analysis/key_frequencies/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed0_arch25k (no analysis/key_frequencies/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed1_arch25k (no analysis/key_frequencies/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed2_arch25k (no analysis/key_frequencies/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed3_arch25k (no analysis/key_frequencies/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed4_arch25k (no analysis/key_frequencies/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed5_arch25k (no analysis/key_frequencies/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed6_arch25k (no analysis/key_frequencies/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed7_arch25k (no analysis/key_frequencies/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed8_arch25k (no analysis/key_frequencies/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed9_arch25k (no analysis/key_frequencies/*.json)

### mlp_mechanism  (41 rows, 11 missing, 19 not applicable to this architecture)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | n_neurons | n_live | structured_n | structured_fraction_of_live |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 512 | 512 | 204 | 0.3984 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step002100 | 2100 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | 512 | 512 | 197 | 0.3848 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 512 | 512 | 153 | 0.2988 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step010850 | 10850 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | 512 | 512 | 158 | 0.3086 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | 512 | 509 | 203 | 0.3988 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 512 | 512 | 451 | 0.8809 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | 512 | 512 | 305 | 0.5957 |
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 512 | 512 | 417 | 0.8145 |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step001725 | 1725 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | 512 | 512 | 205 | 0.4004 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step002125 | 2125 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | 512 | 512 | 192 | 0.375 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 512 | 512 | 136 | 0.2656 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step011475 | 11475 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | 512 | 512 | 148 | 0.2891 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | 512 | 511 | 167 | 0.3268 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 512 | 512 | 445 | 0.8691 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | 512 | 512 | 367 | 0.7168 |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | 512 | 512 | 447 | 0.873 |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 512 | 512 | 211 | 0.4121 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step002175 | 2175 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | 512 | 512 | 199 | 0.3887 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 512 | 512 | 147 | 0.2871 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step010125 | 10125 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | 512 | 512 | 145 | 0.2832 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | 512 | 509 | 168 | 0.3301 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 512 | 512 | 444 | 0.8672 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | 512 | 384 | 215 | 0.5599 |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 512 | 512 | 434 | 0.8477 |
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
| m2h_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp_twohot | 0 | step012775 | 12775 | 0.3 | 1 | no | 25 | 10 | 25000 | 230 | 12775 | 512 | 512 | 223 | 0.4355 |
| m2h_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp_twohot | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 230 | 12775 | 512 | 512 | 349 | 0.6816 |
| m2h_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp_twohot | 1 | step013300 | 13300 | 0.3 | 1 | no | 25 | 10 | 25000 | 240 | 13300 | 512 | 512 | 226 | 0.4414 |

**Missing (11):** m2h_add_p113_wd1.0_frac0.3_seed2_arch25k (no analysis/mlp_mechanism/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed0_arch25k (no analysis/mlp_mechanism/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed1_arch25k (no analysis/mlp_mechanism/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed2_arch25k (no analysis/mlp_mechanism/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed3_arch25k (no analysis/mlp_mechanism/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed4_arch25k (no analysis/mlp_mechanism/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed5_arch25k (no analysis/mlp_mechanism/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed6_arch25k (no analysis/mlp_mechanism/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed7_arch25k (no analysis/mlp_mechanism/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed8_arch25k (no analysis/mlp_mechanism/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed9_arch25k (no analysis/mlp_mechanism/*.json)

### transformer_mechanism  (38 rows, 0 missing, 32 not applicable to this architecture)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | n_neurons | n_live | structured_n | structured_fraction_of_live |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | step000375 | 375 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | 512 | 512 | 274 | 0.5352 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | step000675 | 675 | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | 512 | 512 | 249 | 0.4863 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | step006975 | 6975 | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | 512 | 512 | 263 | 0.5137 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step007975 | 7975 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 512 | 512 | 316 | 0.6172 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | 512 | 512 | 512 | 1 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 512 | 512 | 494 | 0.9648 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | 512 | 512 | 426 | 0.832 |
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | 512 | 512 | 512 | 1 |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | step000675 | 675 | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | 512 | 512 | 288 | 0.5625 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | step000800 | 800 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | 512 | 512 | 354 | 0.6914 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step006250 | 6250 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 512 | 512 | 279 | 0.5449 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | step006500 | 6500 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 512 | 512 | 288 | 0.5625 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 512 | 512 | 506 | 0.9883 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 512 | 512 | 507 | 0.9902 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | 512 | 512 | 443 | 0.8652 |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | 512 | 512 | 498 | 0.9727 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | step000725 | 725 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | 512 | 512 | 326 | 0.6367 |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | step000875 | 875 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | 512 | 512 | 259 | 0.5059 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | step006500 | 6500 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 512 | 512 | 291 | 0.5684 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 512 | 512 | 309 | 0.6035 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 512 | 512 | 509 | 0.9941 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 512 | 512 | 495 | 0.9668 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | 512 | 512 | 494 | 0.9648 |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | 512 | 512 | 491 | 0.959 |
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

### wave_fitting  (40 rows, 12 missing, 19 not applicable to this architecture)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | key_frequencies_used | u_a__fraction_best_aic_sinusoid | u_a__fraction_best_aic_square | u_a__fraction_best_aic_odd_harmonics |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | — | 0.4512 | 0.2891 | 0.2344 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step002100 | 2100 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | — | 0.4434 | 0.2969 | 0.2305 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | — | 0.3516 | 0.375 | 0.2441 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step010850 | 10850 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | — | 0.3965 | 0.3438 | 0.2305 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | — | 0.2871 | 0.373 | 0.3008 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | — | 0.3398 | 0.02734 | 0.627 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | — | 0.1113 | 0.377 | 0.5078 |
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | — | 0.3027 | 0.02148 | 0.6758 |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step001725 | 1725 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | — | 0.3945 | 0.3242 | 0.2559 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step002125 | 2125 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | — | 0.4766 | 0.2754 | 0.2402 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | — | 0.3613 | 0.3906 | 0.2129 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step011475 | 11475 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | — | 0.3496 | 0.3594 | 0.2559 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | — | 0.4297 | 0.2676 | 0.2988 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | — | 0.2539 | 0.01562 | 0.7227 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | — | 0.2305 | 0 | 0.7695 |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | — | 0.2617 | 0.003906 | 0.7344 |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | — | 0.3906 | 0.3125 | 0.2734 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step002175 | 2175 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | — | 0.4551 | 0.2754 | 0.25 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | — | 0.3652 | 0.3691 | 0.2441 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step010125 | 10125 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | — | 0.3652 | 0.3418 | 0.2676 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | — | 0.2461 | 0.2754 | 0.416 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | — | 0.3184 | 0.03711 | 0.6387 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | — | 0.4902 | 0.1543 | 0.3535 |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | — | 0.3887 | 0.007812 | 0.6035 |
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
| m2h_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp_twohot | 0 | step012775 | 12775 | 0.3 | 1 | no | 25 | 10 | 25000 | 230 | 12775 | — | 0.2129 | 0.3594 | 0.4141 |
| m2h_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp_twohot | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 230 | 12775 | — | 0.1289 | 0.209 | 0.6562 |

**Missing (12):** m2h_add_p113_wd1.0_frac0.3_seed1_arch25k (no analysis/wave_fitting/*.json), m2h_add_p113_wd1.0_frac0.3_seed2_arch25k (no analysis/wave_fitting/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed0_arch25k (no analysis/wave_fitting/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed1_arch25k (no analysis/wave_fitting/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed2_arch25k (no analysis/wave_fitting/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed3_arch25k (no analysis/wave_fitting/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed4_arch25k (no analysis/wave_fitting/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed5_arch25k (no analysis/wave_fitting/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed6_arch25k (no analysis/wave_fitting/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed7_arch25k (no analysis/wave_fitting/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed8_arch25k (no analysis/wave_fitting/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed9_arch25k (no analysis/wave_fitting/*.json)

### logit_formula_fit  (77 rows, 12 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | family_size | control_set_size | control_n | sparse_sinusoid__r2_test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 56 | 56 | 50 | 0.4448 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step002100 | 2100 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | 56 | 56 | 50 | 0.4521 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 56 | 56 | 50 | 0.4994 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step010850 | 10850 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | 56 | 56 | 50 | 0.5214 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | 52 | 52 | 50 | 0.7807 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 42 | 42 | 50 | 0.9667 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | 32 | 32 | 50 | 0.985 |
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 35 | 35 | 50 | 0.9876 |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step001725 | 1725 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | 56 | 56 | 50 | 0.4507 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step002125 | 2125 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | 56 | 56 | 50 | 0.4342 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 56 | 56 | 50 | 0.5398 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step011475 | 11475 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | 56 | 56 | 50 | 0.4706 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | 41 | 41 | 50 | 0.9097 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 35 | 35 | 50 | 0.973 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | 28 | 28 | 50 | 0.9902 |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | 29 | 29 | 50 | 0.9895 |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 56 | 56 | 50 | 0.4578 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step002175 | 2175 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | 56 | 56 | 50 | 0.4426 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 56 | 56 | 50 | 0.5104 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step010125 | 10125 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | 56 | 56 | 50 | 0.519 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | 41 | 41 | 50 | 0.7825 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 32 | 32 | 50 | 0.9769 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | 30 | 30 | 50 | 0.9825 |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 31 | 31 | 50 | 0.9855 |
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
| m2h_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp_twohot | 0 | step012775 | 12775 | 0.3 | 1 | no | 25 | 10 | 25000 | 230 | 12775 | 56 | 56 | 50 | 0.1342 |
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | step000375 | 375 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | 11 | 11 | 50 | 0.9046 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | step000675 | 675 | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | 18 | 18 | 50 | 0.8232 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | step006975 | 6975 | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | 19 | 19 | 50 | 0.7632 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step007975 | 7975 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 21 | 21 | 50 | 0.69 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | 16 | 16 | 50 | 0.7028 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 28 | 28 | 50 | 0.4271 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | 16 | 16 | 50 | 0.8537 |
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | 11 | 11 | 50 | 0.6435 |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | step000675 | 675 | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | 24 | 24 | 50 | 0.8386 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | step000800 | 800 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | 11 | 11 | 50 | 0.8493 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step006250 | 6250 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 17 | 17 | 50 | 0.7183 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | step006500 | 6500 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 14 | 14 | 50 | 0.6947 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 14 | 14 | 50 | 0.6708 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 15 | 15 | 50 | 0.695 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | 22 | 22 | 50 | 0.5508 |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | 13 | 13 | 50 | 0.6888 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | step000725 | 725 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | 8 | 8 | 50 | 0.8398 |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | step000875 | 875 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | 25 | 25 | 50 | 0.8233 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | step006500 | 6500 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 24 | 24 | 50 | 0.664 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 19 | 19 | 50 | 0.7433 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 16 | 16 | 50 | 0.8301 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 18 | 18 | 50 | 0.5016 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | 12 | 12 | 50 | 0.4204 |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | 14 | 14 | 50 | 0.6158 |
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

**Missing (12):** m2h_add_p113_wd1.0_frac0.3_seed1_arch25k (no analysis/logit_formula_fit/*.json), m2h_add_p113_wd1.0_frac0.3_seed2_arch25k (no analysis/logit_formula_fit/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed0_arch25k (no analysis/logit_formula_fit/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed1_arch25k (no analysis/logit_formula_fit/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed2_arch25k (no analysis/logit_formula_fit/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed3_arch25k (no analysis/logit_formula_fit/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed4_arch25k (no analysis/logit_formula_fit/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed5_arch25k (no analysis/logit_formula_fit/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed6_arch25k (no analysis/logit_formula_fit/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed7_arch25k (no analysis/logit_formula_fit/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed8_arch25k (no analysis/logit_formula_fit/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed9_arch25k (no analysis/logit_formula_fit/*.json)

### h3_validity  (78 rows, 12 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | h3a__u_a__waveform_sensitivity_top1 | h3a__u_a__waveform_sensitivity_top4 | h3a__u_a__waveform_sensitivity_top8 | h3a__u_a__shape_separation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step002100 | 2100 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step010850 | 10850 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step001725 | 1725 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step002125 | 2125 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step011475 | 11475 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step002175 | 2175 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step010125 | 10125 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
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
| m2h_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp_twohot | 0 | step012775 | 12775 | 0.3 | 1 | no | 25 | 10 | 25000 | 230 | 12775 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| m2h_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp_twohot | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 230 | 12775 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | step000375 | 375 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | step000675 | 675 | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | step006975 | 6975 | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step007975 | 7975 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | step000675 | 675 | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | step000800 | 800 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step006250 | 6250 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | step006500 | 6500 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | step000725 | 725 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | step000875 | 875 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | step006500 | 6500 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | 0.1893 | 0.05012 | 0.0248 | 0.1387 |
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

**Missing (12):** m2h_add_p113_wd1.0_frac0.3_seed1_arch25k (no analysis/h3_validity/*.json), m2h_add_p113_wd1.0_frac0.3_seed2_arch25k (no analysis/h3_validity/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed0_arch25k (no analysis/h3_validity/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed1_arch25k (no analysis/h3_validity/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed2_arch25k (no analysis/h3_validity/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed3_arch25k (no analysis/h3_validity/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed4_arch25k (no analysis/h3_validity/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed5_arch25k (no analysis/h3_validity/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed6_arch25k (no analysis/h3_validity/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed7_arch25k (no analysis/h3_validity/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed8_arch25k (no analysis/h3_validity/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed9_arch25k (no analysis/h3_validity/*.json)

### causal_ablation  (76 rows, 13 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | baseline_test_acc | n_structured | g4_passed | g4_remove_structured_necessary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 0.9502 | 204 | no | no |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step002100 | 2100 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | 0.954 | 197 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step009125 | 9125 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 0.9513 | 128 | no | no |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step010850 | 10850 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | 0.9517 | 158 | no | no |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | 1 | 203 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 1 | 425 | no | no |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | 1 | 305 | no | no |
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 1 | 417 | no | no |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step001725 | 1725 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | 0.9551 | 205 | no | no |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step002125 | 2125 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | 0.9576 | 192 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step010075 | 10075 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 0.9508 | 110 | no | no |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step011475 | 11475 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | 0.9528 | 148 | no | no |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | 1 | 167 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 1 | 434 | no | no |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | 1 | 367 | no | no |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | 1 | 447 | no | no |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step001700 | 1700 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 0.9568 | 211 | no | no |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step002175 | 2175 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | 0.958 | 199 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step009175 | 9175 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 0.9512 | 123 | no | no |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step010125 | 10125 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | 0.9528 | 145 | no | no |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | 1 | 168 | yes | yes |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 1 | 418 | no | no |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | 1 | 215 | no | no |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 1 | 434 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step008150 | 8150 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 0.9541 | 98 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 1 | 489 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step008650 | 8650 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 0.9512 | 108 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 1 | 464 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step009375 | 9375 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 0.9507 | 129 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 1 | 436 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step009900 | 9900 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 0.9503 | 120 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 1 | 427 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step009200 | 9200 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 0.9537 | 123 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 1 | 448 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step009800 | 9800 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 0.9509 | 113 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 1 | 458 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step009300 | 9300 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 0.9508 | 112 | no | no |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 1 | 459 | no | no |
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | step000375 | 375 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | 0.9681 | 274 | no | no |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | step000675 | 675 | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | 0.9518 | 249 | no | no |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | step006975 | 6975 | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | 0.9511 | 263 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step007975 | 7975 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 0.9503 | 247 | no | no |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | 1 | 512 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 0.9977 | 487 | no | no |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | 1 | 426 | no | no |
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | 1 | 512 | no | no |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | step000675 | 675 | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | 0.9519 | 288 | no | no |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | step000800 | 800 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | 0.9604 | 354 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step006250 | 6250 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 0.9505 | 219 | no | no |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | step006500 | 6500 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 0.9547 | 288 | no | no |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 0.9978 | 506 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 0.9994 | 504 | no | no |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | 0.992 | 443 | no | no |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | 0.9986 | 498 | no | no |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | step000725 | 725 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | 0.9543 | 326 | no | no |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | step000875 | 875 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | 0.9527 | 259 | no | no |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | step006500 | 6500 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 0.9588 | 291 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.9549 | 244 | no | no |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 1 | 509 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.9971 | 491 | no | no |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | 1 | 494 | no | no |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | step025000 | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | 1 | 491 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step007425 | 7425 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 0.9502 | 257 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 0.9996 | 508 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step010275 | 10275 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 0.9523 | 213 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 1 | 497 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step005625 | 5625 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 0.9526 | 188 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 1 | 512 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step006325 | 6325 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 0.9531 | 216 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 1 | 504 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step007750 | 7750 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 0.9516 | 216 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 1 | 499 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step008125 | 8125 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.954 | 246 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0.9993 | 503 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step005825 | 5825 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 0.9512 | 247 | no | no |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | step025000 | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 0.9998 | 494 | no | no |

**Missing (13):** m2h_add_p113_wd1.0_frac0.3_seed0_arch25k (no analysis/causal_ablation/*.json), m2h_add_p113_wd1.0_frac0.3_seed1_arch25k (no analysis/causal_ablation/*.json), m2h_add_p113_wd1.0_frac0.3_seed2_arch25k (no analysis/causal_ablation/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed0_arch25k (no analysis/causal_ablation/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed1_arch25k (no analysis/causal_ablation/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed2_arch25k (no analysis/causal_ablation/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed3_arch25k (no analysis/causal_ablation/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed4_arch25k (no analysis/causal_ablation/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed5_arch25k (no analysis/causal_ablation/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed6_arch25k (no analysis/causal_ablation/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed7_arch25k (no analysis/causal_ablation/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed8_arch25k (no analysis/causal_ablation/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed9_arch25k (no analysis/causal_ablation/*.json)

### structure_over_time  (38 rows, 13 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | role__init | role__pre_memorization | role__memorization | role__mid_plateau |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | all_checkpoints | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | 0 | 0 | 170 | 6000 |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | all_checkpoints | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | 0 | 0 | 200 | 1000 |
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | all_checkpoints | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 0 | 0 | 190 | 1000 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | all_checkpoints | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | 0 | 0 | 170 | 6000 |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | all_checkpoints | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | 0 | 0 | 210 | 1000 |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | all_checkpoints | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | 0 | 0 | 190 | 1000 |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | all_checkpoints | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | 0 | 0 | 170 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | all_checkpoints | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | 0 | 0 | 200 | 1000 |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | all_checkpoints | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | 0 | 0 | 190 | 1000 |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | 0 | 0 | 160 | 4000 |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | 0 | 0 | 160 | 4000 |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | 0 | 0 | 160 | 5000 |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | 0 | 0 | 160 | 5000 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | all_checkpoints | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | 0 | 0 | 150 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | all_checkpoints | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | 0 | 0 | 170 | 500 |
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | all_checkpoints | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | 0 | 0 | 140 | 500 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | all_checkpoints | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 0 | 0 | 170 | 3000 |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | 0 | 0 | 140 | 3000 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | all_checkpoints | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | 0 | 0 | 200 | 500 |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | all_checkpoints | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | 0 | 0 | 150 | 500 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | all_checkpoints | 25000 | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | 0 | 0 | 170 | 3000 |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | all_checkpoints | 25000 | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | 0 | 0 | 200 | 500 |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | all_checkpoints | 25000 | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | 0 | 0 | 140 | 500 |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | 0 | 0 | 150 | 5000 |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | 0 | 0 | 140 | 3000 |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | 0 | 0 | 140 | 3000 |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | 0 | 0 | 140 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | all_checkpoints | 25000 | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | 0 | 0 | 140 | 3000 |

**Missing (13):** m2h_add_p113_wd1.0_frac0.3_seed0_arch25k (no analysis/structure_over_time/*.json), m2h_add_p113_wd1.0_frac0.3_seed1_arch25k (no analysis/structure_over_time/*.json), m2h_add_p113_wd1.0_frac0.3_seed2_arch25k (no analysis/structure_over_time/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed0_arch25k (no analysis/structure_over_time/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed1_arch25k (no analysis/structure_over_time/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed2_arch25k (no analysis/structure_over_time/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed3_arch25k (no analysis/structure_over_time/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed4_arch25k (no analysis/structure_over_time/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed5_arch25k (no analysis/structure_over_time/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed6_arch25k (no analysis/structure_over_time/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed7_arch25k (no analysis/structure_over_time/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed8_arch25k (no analysis/structure_over_time/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed9_arch25k (no analysis/structure_over_time/*.json)

### progress_measures  (38 rows, 13 missing)

| run_id | arch | seed | tag | step | train_frac | weight_decay | grokfast | eval_every_test | eval_every_train | steps_completed | memorization_step | generalization_step | legacy_broad_mask__restricted | legacy_broad_mask__excluded | nanda_exact__restricted | nanda_exact__excluded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | mlp | 0 | all_checkpoints | all_checkpoints | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10850 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_seed0_arch25k | mlp | 0 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9125 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | mlp | 0 | all_checkpoints | all_checkpoints | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2100 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.5_seed0_arch25k | mlp | 0 | all_checkpoints | all_checkpoints | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | mlp | 1 | all_checkpoints | all_checkpoints | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 11475 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_seed1_arch25k | mlp | 1 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 10075 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | mlp | 1 | all_checkpoints | all_checkpoints | 0.5 | 1 | yes | 25 | 10 | 25000 | 210 | 2125 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.5_seed1_arch25k | mlp | 1 | all_checkpoints | all_checkpoints | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1725 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | mlp | 2 | all_checkpoints | all_checkpoints | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 10125 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_seed2_arch25k | mlp | 2 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9175 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | mlp | 2 | all_checkpoints | all_checkpoints | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 2175 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.5_seed2_arch25k | mlp | 2 | all_checkpoints | all_checkpoints | 0.5 | 1 | no | 25 | 10 | 25000 | 190 | 1700 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_seed3_arch25k | mlp | 3 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8150 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_seed4_arch25k | mlp | 4 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 8650 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_seed5_arch25k | mlp | 5 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9375 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_seed6_arch25k | mlp | 6 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9900 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_seed7_arch25k | mlp | 7 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9200 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_seed8_arch25k | mlp | 8 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9800 | — | — | — | — |
| mlp_add_p113_wd1.0_frac0.3_seed9_arch25k | mlp | 9 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 160 | 9300 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k | transformer | 0 | all_checkpoints | all_checkpoints | 0.3 | 1 | yes | 25 | 10 | 25000 | 150 | 6975 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_seed0_arch25k | transformer | 0 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7975 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k | transformer | 0 | all_checkpoints | all_checkpoints | 0.5 | 1 | yes | 25 | 10 | 25000 | 170 | 675 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.5_seed0_arch25k | transformer | 0 | all_checkpoints | all_checkpoints | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 375 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k | transformer | 1 | all_checkpoints | all_checkpoints | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_seed1_arch25k | transformer | 1 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6250 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k | transformer | 1 | all_checkpoints | all_checkpoints | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 800 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.5_seed1_arch25k | transformer | 1 | all_checkpoints | all_checkpoints | 0.5 | 1 | no | 25 | 10 | 25000 | 150 | 675 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k | transformer | 2 | all_checkpoints | all_checkpoints | 0.3 | 1 | yes | 25 | 10 | 25000 | 170 | 6500 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_seed2_arch25k | transformer | 2 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k | transformer | 2 | all_checkpoints | all_checkpoints | 0.5 | 1 | yes | 25 | 10 | 25000 | 200 | 725 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.5_seed2_arch25k | transformer | 2 | all_checkpoints | all_checkpoints | 0.5 | 1 | no | 25 | 10 | 25000 | 140 | 875 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_seed3_arch25k | transformer | 3 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7425 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_seed4_arch25k | transformer | 4 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 150 | 10275 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_seed5_arch25k | transformer | 5 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5625 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_seed6_arch25k | transformer | 6 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 6325 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_seed7_arch25k | transformer | 7 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 7750 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_seed8_arch25k | transformer | 8 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 8125 | — | — | — | — |
| txf_add_p113_wd1.0_frac0.3_seed9_arch25k | transformer | 9 | all_checkpoints | all_checkpoints | 0.3 | 1 | no | 25 | 10 | 25000 | 140 | 5825 | — | — | — | — |

**Missing (13):** m2h_add_p113_wd1.0_frac0.3_seed0_arch25k (no analysis/progress_measures/*.json), m2h_add_p113_wd1.0_frac0.3_seed1_arch25k (no analysis/progress_measures/*.json), m2h_add_p113_wd1.0_frac0.3_seed2_arch25k (no analysis/progress_measures/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed0_arch25k (no analysis/progress_measures/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed1_arch25k (no analysis/progress_measures/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed2_arch25k (no analysis/progress_measures/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed3_arch25k (no analysis/progress_measures/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed4_arch25k (no analysis/progress_measures/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed5_arch25k (no analysis/progress_measures/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed6_arch25k (no analysis/progress_measures/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed7_arch25k (no analysis/progress_measures/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed8_arch25k (no analysis/progress_measures/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed9_arch25k (no analysis/progress_measures/*.json)

### bounded_alternative  (20 rows, 31 missing)

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

**Missing (31):** m2h_add_p113_wd1.0_frac0.3_seed0_arch25k (no analysis/bounded_alternative/*.json), m2h_add_p113_wd1.0_frac0.3_seed1_arch25k (no analysis/bounded_alternative/*.json), m2h_add_p113_wd1.0_frac0.3_seed2_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed0_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed1_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed2_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed3_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed4_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed5_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed6_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed7_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed8_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_dm572_seed9_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.5_seed0_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.5_seed1_arch25k (no analysis/bounded_alternative/*.json), mlp_add_p113_wd1.0_frac0.5_seed2_arch25k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_gf2.0_seed0_arch25k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_gf2.0_seed1_arch25k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.3_gf2.0_seed2_arch25k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.5_gf2.0_seed0_arch25k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.5_gf2.0_seed1_arch25k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.5_gf2.0_seed2_arch25k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.5_seed0_arch25k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.5_seed1_arch25k (no analysis/bounded_alternative/*.json), txf_add_p113_wd1.0_frac0.5_seed2_arch25k (no analysis/bounded_alternative/*.json)
