# Convergence-extension runbook (S1) — external compute on EC2

**Status: proposal / hand-off document. Not a result.**

This document describes how the **convergence-extension block (S1)** is executed on AWS EC2
by an external contributor (Jana Stucke, mentor) because of a submission deadline, and how the
resulting artifacts are handed back to the student authors.

It changes **no** existing result. The 25,000-step primary block stays exactly as reported in
`RESULTS.md`. S1 is a **new block** that answers one open question the study already documents
against itself:

> `docs/LIMITATIONS.md` B3 — the MLP's `structured_fraction_of_live` is settled in only **2 of 10**
> seeds at the 25,000-step budget and is still rising, while the transformer's is settled 10/10.
> A longer budget would be expected to **shrink** the reported architecture gap.

S1 measures whether it does.

---

## 0. Honesty and authorship conditions (read first)

These are binding for the artifacts produced here.

1. **The runs are external compute, and are disclosed as such.** Add the row in §6 to
   `AI_DISCLOSURE.md`. Do not describe these runs as the students' own execution.
2. **Interpretation stays with the student authors.** This document, and the artifacts it
   produces, contain measurements only. `RESULTS.md`, `LIMITATIONS.md` and the final
   interpretation remain human-authored by the students, per `docs/HUMAN_DECISIONS.md` §F and
   the existing `docs/HUMAN_INTERPRETATION_TEMPLATE.md`.
3. **Reproduction before publication.** The students should re-run at least one seed pair on
   their own hardware and confirm it matches before any S1 number enters a submission. The
   configuration is byte-identical to theirs, so this is a check, not a re-derivation.
4. **Competition eligibility is not assessed here.** Whether externally-executed compute is
   permitted is a rules question for the student authors to confirm. This document does not
   evaluate it.
5. **No threshold, definition or pass rule is changed.** S1 changes exactly one variable, the
   step budget. Everything else is the frozen configuration.

---

## 1. Working directory and branch

Local working copy (fresh clone of the students' updated branch). `$REPO` below is the
clone's root on the machine doing the work; set it once and the rest of this document is
machine-independent:

```bash
export REPO="$HOME/<path-to>/GROKVERSE-MP"
```

Code root inside it:

```
$REPO/Claude/grokverse
```

Branch for this work, already created, **never push to `main` or `GROKVERSE-MP`**:

```
arch-study-convergence
```

Push with an explicit branch name:

```bash
cd "$REPO"
git push -u origin arch-study-convergence
```

---

## 2. What S1 runs, and why these settings

Identical to the frozen primary block except `--steps`. Verified against
`docs/PREREGISTRATION.md` §2.1 and `train.py`'s CLI.

| setting | value | source |
|---|---|---|
| task | `(a + b) mod 113` | frozen |
| `--train-frac` | `0.3` | frozen |
| `--weight-decay` | `1.0` | frozen |
| Grokfast | off | frozen (primary block has none) |
| `--threads` | `1` | frozen, and required for the parallel plan |
| seeds | `0..9`, both architectures | frozen, paired |
| `--steps` | **`100000`** | **the only change** |
| `--study` | `conv100k` | new run_id suffix, keeps artifacts separate |

**Stopping rule, fixed before the run:** stop at 100,000 steps. Convergence is then *assessed*
with the study's existing rule (a metric counts as settled if it changes < 5 % between the last
two checkpoints, `docs/HUMAN_DECISIONS.md` B9). We do not stop early on a metric, because that
would make the budget outcome-dependent.

---

## 3. Instance choice

The workload is 20 independent single-threaded CPU runs. It is embarrassingly parallel and
**gains nothing from a GPU** — the models are 226k / 204k parameters, full-batch, so a GPU would
be dominated by kernel-launch overhead.

Measured base rates from the students' own `PROGRESS.md`: **4.05 h** per transformer run and
**~0.9 h** per MLP run at 25,000 steps, single-threaded. At 100,000 steps (4×):

- transformer: ~16.2 h/run × 10 = 162 CPU-h
- MLP: ~3.6 h/run × 10 = 36 CPU-h
- **total ≈ 198 CPU-hours**

| instance | vCPU | est. wall clock | notes |
|---|---|---|---|
| `c7i.8xlarge` | 32 | ~7 h | cheapest sensible option |
| **`c7i.16xlarge`** | **64** | **~3.5–4 h** | **recommended**: 20 runs fit in one wave |
| `c7i.24xlarge` | 96 | ~3.5 h | no real gain, only 20 runs |

**Recommended: `c7i.16xlarge`**, compute-optimised, 64 vCPU. All 20 runs launch at once, each
pinned to one thread, and the wall clock is set by the slowest single run (~16 h transformer)…
**except** that with 64 cores you can also split each architecture's seeds across cores without
contention, so the real limit is the 16.2 h transformer run.

> **Reality check on wall clock:** because a single transformer run is inherently ~16 h
> single-threaded, no amount of parallelism makes S1 finish faster than **~16–17 hours** unless
> the step budget is reduced. The 3.5 h figures above are *aggregate CPU throughput*, not
> completion time. Plan for **one overnight run (~17 h)** on `c7i.16xlarge`.
> If that is too long for the deadline, use `--steps 50000` instead: ~8.1 h per transformer run,
> which still doubles the original budget and directly tests the B3 concern.

Storage: 100 GB gp3. Checkpoints are 21 per run × 20 runs; the existing primary block's
artifacts are the size guide, and 100 GB is comfortable headroom.

---

## 4. Launch, setup, run

### 4.1 Resolve a current AMI and launch

```bash
export AWS_REGION=eu-central-1

AMI=$(aws ssm get-parameters --region $AWS_REGION \
  --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameters[0].Value' --output text)
echo "AMI=$AMI"

aws ec2 run-instances --region $AWS_REGION \
  --image-id "$AMI" \
  --instance-type c7i.16xlarge \
  --iam-instance-profile Name=<YOUR_SSM_S3_INSTANCE_PROFILE> \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=grokverse-conv100k}]' \
  --instance-initiated-shutdown-behavior terminate \
  --query 'Instances[0].InstanceId' --output text
```

The instance profile needs `AmazonSSMManagedInstanceCore` (so you get Session Manager with no
SSH key and no open inbound port) plus write access to your S3 bucket.

`--instance-initiated-shutdown-behavior terminate` means the runner script can shut the box down
when finished and it self-terminates, so a finished job cannot keep billing.

### 4.2 Connect

```bash
aws ssm start-session --region $AWS_REGION --target <INSTANCE_ID>
```

### 4.3 Set up

```bash
sudo dnf install -y git python3.12 python3.12-pip tar gzip
cd /home/ssm-user
git clone --branch arch-study-convergence \
  https://github.com/HenrikBrehm/GROKVERSE.git grokverse-work
cd grokverse-work/Claude/grokverse/training

python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt      # torch 2.12.1+cpu, numpy 2.4.6, matplotlib 3.11.0

python test_core.py                  # sanity gate before spending compute
```

If `torch==2.12.1+cpu` will not resolve from PyPI directly, add the CPU index:

```bash
pip install torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4.4 Run all 20 in parallel, one thread each

```bash
cd /home/ssm-user/grokverse-work/Claude/grokverse/training
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p logs
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for arch in transformer mlp; do
    nohup python -m grokverse.train \
      --config nanda --arch $arch --task add \
      --train-frac 0.3 --weight-decay 1.0 \
      --steps 100000 --seed $seed --threads 1 \
      --study conv100k \
      > logs/${arch}_seed${seed}.log 2>&1 &
  done
done
jobs -l | wc -l        # expect 20
```

Monitor:

```bash
grep -h 'step' logs/transformer_seed0.log | tail -3
ls -1 runs/ | grep conv100k | wc -l
```

### 4.5 Analysis over the new block

```bash
python -m grokverse.analysis.driver --runs "*_conv100k"
python -m grokverse.analysis.aggregate \
  --runs "txf_add_p113_wd1.0_frac0.3_seed?_conv100k" \
         "mlp_add_p113_wd1.0_frac0.3_seed?_conv100k" \
  --out results/aggregate_conv100k
python -m grokverse.analysis.decision_tree     --aggregate-dir results/aggregate_conv100k --point final
python -m grokverse.analysis.statistics_report --aggregate-dir results/aggregate_conv100k
python -m grokverse.analysis.h3_report         --aggregate-dir results/aggregate_conv100k
python -m grokverse.analysis.h4_report         --aggregate-dir results/aggregate_conv100k
```

> Verify the exact glob against `run_manifest.csv` naming before relying on it; the primary block
> uses `..._arch25k`, so `..._conv100k` follows the same pattern via `--study`.

---

## 5. What to save to S3

Create the bucket once (private, versioned):

```bash
aws s3api create-bucket --bucket grokverse-conv100k-<ACCOUNT_ID> \
  --region eu-central-1 \
  --create-bucket-configuration LocationConstraint=eu-central-1
aws s3api put-public-access-block --bucket grokverse-conv100k-<ACCOUNT_ID> \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-versioning --bucket grokverse-conv100k-<ACCOUNT_ID> \
  --versioning-configuration Status=Enabled
```

**Save these, in this order of importance:**

| what | path | why |
|---|---|---|
| `run.json` per run | `runs/*_conv100k/run.json` | curves, transitions, config, git commit, split hash — the scientific record |
| `manifest.json` per run | `runs/*_conv100k/manifest.json` | provenance: versions, platform, parameter counts, weight norms |
| `checkpoints.json` | `runs/*_conv100k/checkpoints.json` | checkpoint index with SHA-256 per file |
| `embeddings.npy` | `runs/*_conv100k/embeddings.npy` | small, needed for Fourier analysis |
| all analysis artifacts | `results/` | the measured outputs and aggregate tables |
| `run_manifest.csv` | `results/run_manifest.csv` | the 20-run index |
| `train.log` per run | `logs/` | audit trail |
| **final checkpoint only** | `runs/*_conv100k/ckpt_step100000.pt` | enough to re-run any ablation |

```bash
cd /home/ssm-user/grokverse-work/Claude/grokverse/training
B=s3://grokverse-conv100k-<ACCOUNT_ID>/conv100k

# light, always
aws s3 cp results/ $B/results/ --recursive
aws s3 cp logs/    $B/logs/    --recursive
for d in runs/*_conv100k; do
  aws s3 cp "$d" "$B/$d/" --recursive \
    --exclude "*" \
    --include "run.json" --include "manifest.json" \
    --include "checkpoints.json" --include "embeddings.npy" \
    --include "ckpt_step100000.pt"
done

# optional: full checkpoint history (large)
# for d in runs/*_conv100k; do aws s3 cp "$d" "$B/full/$d/" --recursive; done
```

**Do not** upload the `.venv`.

### 5.1 Teardown

```bash
aws ec2 terminate-instances --region eu-central-1 --instance-ids <INSTANCE_ID>
aws ec2 describe-instances  --region eu-central-1 --instance-ids <INSTANCE_ID> \
  --query 'Reservations[0].Instances[0].State.Name' --output text
```

Confirm it says `shutting-down` or `terminated`. Do not rely on the shutdown behaviour alone.

---

## 6. Documentation to add alongside the results

When the artifacts come back, add these. **Nothing else in the repo changes.**

**a) `AI_DISCLOSURE.md`** — new row:

| item | who |
|---|---|
| Convergence-extension block S1 (20 runs at 100,000 steps) executed on external AWS EC2 compute by the mentor, using the students' frozen configuration and code at branch `arch-study-convergence`; artifacts handed back unmodified | External (mentor), disclosed; interpretation by the student authors |

**b) `docs/LABBOOK.md`** — one append-only entry naming: the date, the branch and commit, the
instance type, the single changed variable (`--steps 100000`), the S3 location, and the fact that
no threshold moved.

**c) `PROGRESS.md`** — one block in the existing format: Done / Measured results / Verification /
Open questions / Next.

**d) `docs/LIMITATIONS.md` B3** — append the S1 outcome. B3 currently predicts a longer budget
would shrink the gap. Record what actually happened, including if it did not.

**e) `RESULTS.md`** — a **new** section, not an edit of the existing numbers, reporting S1 as a
separate block. The 25,000-step results stay as they are, labelled as budget-fixed.

**f) This file** — kept as the record of how the compute was done.

---

## 7. What S1 can and cannot settle

**Can:** whether the MLP's structured fraction converges given 4× the budget; whether the
+0.085 architecture gap shrinks, holds, or grows; whether the unsettled waveform shares
(the transformer's sinusoid share moved 0.255 between steps 20k and 25k) stabilise; whether the
G1 margin changes for the MLP.

**Cannot:** rescue G4. `docs/HUMAN_DECISIONS.md` D6 already measured that *every* available
structured-neuron definition selects 86–100 % of live neurons, and notes a longer budget would push
the MLP's fraction **up**, making the set larger and the size-matched ablation *less* able to
discriminate. G4 needs a different **kind** of definition, which is spec S2 (ranked graded
ablation), not a longer budget.

So S1 closes the convergence loophole. It does not by itself produce the "they learn differently,
and here is why" claim; that needs S2, S3 and S4 as well.
