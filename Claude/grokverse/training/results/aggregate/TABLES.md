# Aggregated analysis tables

Generated 2026-09-04T02:20:05.902264+00:00 at commit `e386f59fd9b85e5db66361e9084d8e674137a24f` over 20 runs matching `*_frac0.3_seed*_arch25k`.

Every row is one run at one measurement point. **All seeds are shown** — no row is averaged away (master prompt §16). Numbers are copied from the per-run files named in `source_file`; nothing here is recomputed.

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
