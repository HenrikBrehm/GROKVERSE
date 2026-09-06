# Graded structured-neuron ablation

Pre-registered in `docs/PREREGISTRATION.md` §14; human decision `docs/HUMAN_DECISIONS.md` **D7** is open. **Measurement only — this feeds no gate criterion.** The evidence gate stands at `neither_passes` and §14.5 fixes in advance that no outcome here reopens G4.

Rows: 40 run-checkpoints from the 20 primary runs.

## 1. Graded structured ablation — primary score (§14.3)

| architecture | checkpoint | fraction | n selected | seeds discriminable | passes 8/10 | median drop | median control drop |
|---|---|---|---|---|---|---|---|
| mlp | crossing | 0.01 | 5 | 10/10 | **yes** | 0.1867 | 0.0101 |
| mlp | crossing | 0.02 | 10 | 10/10 | **yes** | 0.3535 | 0.0207 |
| mlp | crossing | 0.05 | 26 | 10/10 | **yes** | 0.6792 | 0.0708 |
| mlp | crossing | 0.10 | 51 | 10/10 | **yes** | 0.8865 | 0.1426 |
| mlp | crossing | 0.25 | 128 | 10/10 | **yes** | 0.9494 | 0.3801 |
| mlp | crossing | 0.50 | 256 | 9/10 | **yes** | 0.9498 | 0.7103 |
| mlp | final | 0.01 | 5 | 0/10 | no | 0.0000 | 0.0000 |
| mlp | final | 0.02 | 10 | 0/10 | no | 0.0000 | 0.0000 |
| mlp | final | 0.05 | 26 | 0/10 | no | 0.0000 | 0.0000 |
| mlp | final | 0.10 | 51 | 0/10 | no | 0.0000 | 0.0000 |
| mlp | final | 0.25 | 128 | 7/10 | no | 0.0021 | 0.0002 |
| mlp | final | 0.50 | 256 | 9/10 | **yes** | 0.4116 | 0.0522 |
| transformer | crossing | 0.01 | 5 | 9/10 | **yes** | 0.0312 | 0.0035 |
| transformer | crossing | 0.02 | 10 | 10/10 | **yes** | 0.0998 | 0.0070 |
| transformer | crossing | 0.05 | 26 | 10/10 | **yes** | 0.2782 | 0.0205 |
| transformer | crossing | 0.10 | 51 | 10/10 | **yes** | 0.4987 | 0.0548 |
| transformer | crossing | 0.25 | 128 | 10/10 | **yes** | 0.7793 | 0.2071 |
| transformer | crossing | 0.50 | 256 | 10/10 | **yes** | 0.9062 | 0.5410 |
| transformer | final | 0.01 | 5 | 2/10 | no | 0.0000 | 0.0000 |
| transformer | final | 0.02 | 10 | 6/10 | no | 0.0017 | 0.0000 |
| transformer | final | 0.05 | 26 | 9/10 | **yes** | 0.0958 | 0.0002 |
| transformer | final | 0.10 | 51 | 9/10 | **yes** | 0.2100 | 0.0004 |
| transformer | final | 0.25 | 128 | 10/10 | **yes** | 0.4183 | 0.0048 |
| transformer | final | 0.50 | 256 | 10/10 | **yes** | 0.7888 | 0.0897 |

- **mlp, crossing:** smallest discriminating fraction: **0.01**
- **mlp, final:** smallest discriminating fraction: **0.50**
- **transformer, crossing:** smallest discriminating fraction: **0.01**
- **transformer, final:** smallest discriminating fraction: **0.05**

## 2. Sensitivity score (mean of the three family fractions)

| architecture | checkpoint | fraction | n selected | seeds discriminable | passes 8/10 | median drop | median control drop |
|---|---|---|---|---|---|---|---|
| mlp | crossing | 0.01 | 5 | 10/10 | **yes** | 0.1908 | 0.0101 |
| mlp | crossing | 0.02 | 10 | 10/10 | **yes** | 0.3667 | 0.0207 |
| mlp | crossing | 0.05 | 26 | 10/10 | **yes** | 0.7078 | 0.0708 |
| mlp | crossing | 0.10 | 51 | 10/10 | **yes** | 0.8981 | 0.1426 |
| mlp | crossing | 0.25 | 128 | 10/10 | **yes** | 0.9491 | 0.3801 |
| mlp | crossing | 0.50 | 256 | 9/10 | **yes** | 0.9500 | 0.7103 |
| mlp | final | 0.01 | 5 | 0/10 | no | 0.0000 | 0.0000 |
| mlp | final | 0.02 | 10 | 0/10 | no | 0.0000 | 0.0000 |
| mlp | final | 0.05 | 26 | 0/10 | no | 0.0000 | 0.0000 |
| mlp | final | 0.10 | 51 | 0/10 | no | 0.0000 | 0.0000 |
| mlp | final | 0.25 | 128 | 5/10 | no | 0.0018 | 0.0002 |
| mlp | final | 0.50 | 256 | 10/10 | **yes** | 0.4235 | 0.0522 |
| transformer | crossing | 0.01 | 5 | 9/10 | **yes** | 0.0373 | 0.0035 |
| transformer | crossing | 0.02 | 10 | 10/10 | **yes** | 0.0932 | 0.0070 |
| transformer | crossing | 0.05 | 26 | 10/10 | **yes** | 0.2585 | 0.0205 |
| transformer | crossing | 0.10 | 51 | 10/10 | **yes** | 0.5064 | 0.0548 |
| transformer | crossing | 0.25 | 128 | 10/10 | **yes** | 0.7836 | 0.2071 |
| transformer | crossing | 0.50 | 256 | 10/10 | **yes** | 0.9094 | 0.5410 |
| transformer | final | 0.01 | 5 | 0/10 | no | 0.0002 | 0.0000 |
| transformer | final | 0.02 | 10 | 3/10 | no | 0.0002 | 0.0000 |
| transformer | final | 0.05 | 26 | 8/10 | **yes** | 0.0220 | 0.0002 |
| transformer | final | 0.10 | 51 | 9/10 | **yes** | 0.1351 | 0.0004 |
| transformer | final | 0.25 | 128 | 10/10 | **yes** | 0.4921 | 0.0048 |
| transformer | final | 0.50 | 256 | 10/10 | **yes** | 0.8531 | 0.0897 |

- **mlp, crossing:** smallest discriminating fraction: **0.01**
- **mlp, final:** smallest discriminating fraction: **0.50**
- **transformer, crossing:** smallest discriminating fraction: **0.01**
- **transformer, final:** smallest discriminating fraction: **0.05**

## 3. The IPR sweep that already existed (D2) — read at matching fractions, no new computation

| architecture | checkpoint | fraction | n selected | seeds discriminable | passes 8/10 | median drop | median control drop |
|---|---|---|---|---|---|---|---|
| mlp | crossing | 0.05 | — | 10/10 | **yes** | 0.6811 | 0.0699 |
| mlp | crossing | 0.10 | — | 10/10 | **yes** | 0.8962 | 0.1453 |
| mlp | crossing | 0.25 | — | 10/10 | **yes** | 0.9491 | 0.3810 |
| mlp | crossing | 0.50 | — | 9/10 | **yes** | 0.9500 | 0.7077 |
| mlp | final | 0.05 | — | 0/10 | no | 0.0000 | 0.0000 |
| mlp | final | 0.10 | — | 1/10 | no | 0.0000 | 0.0000 |
| mlp | final | 0.25 | — | 8/10 | **yes** | 0.0183 | 0.0001 |
| mlp | final | 0.50 | — | 10/10 | **yes** | 0.6099 | 0.0484 |
| transformer | crossing | 0.05 | — | 10/10 | **yes** | 0.4358 | 0.0211 |
| transformer | crossing | 0.10 | — | 10/10 | **yes** | 0.6466 | 0.0510 |
| transformer | crossing | 0.25 | — | 10/10 | **yes** | 0.7880 | 0.1917 |
| transformer | crossing | 0.50 | — | 10/10 | **yes** | 0.9081 | 0.5318 |
| transformer | final | 0.05 | — | 9/10 | **yes** | 0.0180 | 0.0001 |
| transformer | final | 0.10 | — | 10/10 | **yes** | 0.1397 | 0.0004 |
| transformer | final | 0.25 | — | 10/10 | **yes** | 0.5382 | 0.0053 |
| transformer | final | 0.50 | — | 10/10 | **yes** | 0.8534 | 0.0826 |

- **mlp, crossing:** smallest discriminating fraction: **0.05**
- **mlp, final:** smallest discriminating fraction: **0.25**
- **transformer, crossing:** smallest discriminating fraction: **0.05**
- **transformer, final:** smallest discriminating fraction: **0.05**

