# Verdicts — eval, profiling & the Policy Autopsy

The [doctors](doctors.md) judge the *inputs* to learning — assets, datasets,
trajectories. The verdict stack judges the *output*: a trained policy and the
deploy loop it runs in. Four tools, all in the
[`caliper_learn` sidecar](learning.md), each answering one question nobody
instruments until it is too late:

| Tool | Question it answers | Codes |
|---|---|---|
| **Eval harness** (`caliper_learn.eval`) | does the policy actually solve the task, or did the loss just go down? | `E001`–`E003` |
| **Latency profiler** (`caliper_learn.profile`) | can the deploy loop honestly hold the requested control rate? | `L001`–`L003` |
| **Policy debugger** (`caliper_learn.debugger`) | *why* does the trained policy do nothing? | `P001`–`P008` |
| **Autopsy** (`caliper_learn.autopsy`) | all of the above, one report, one verdict | `D` + `P` + `E` + `L` |

They follow the doctors' shared contract: findings are **data, not errors**
(the report is the product), every check has a **stable code** you can filter
on in JSON, every finding carries a plain-English `message` naming the
consequence and a `fix_hint` saying what to do, sorted most-severe-first.
Everything except wall-clock timing is deterministic: the same
(checkpoint, dataset, task, seed) produces byte-identical serialized output —
the tests assert exact equality.

Where to run them:

| | `caliper` CLI (Rust) | `caliper-learn` CLI | Python | Studio |
|---|---|---|---|---|
| Eval / sweep | ✗ ¹ | `caliper-learn eval` | `evaluate` / `sweep` | ✗ ¹ |
| Latency profile | ✗ ¹ | `caliper-learn profile` | `profile_rollout` | ✗ ¹ |
| Policy debugger | ✗ ¹ | `caliper-learn debug` | `analyze_policy` | ✗ ¹ |
| Autopsy | ✗ ¹ | `caliper-learn autopsy` | `autopsy` | ✗ ¹ |

¹ Policy inference is Python-side (torch + the safetensors-only `hub` loader),
so neither the Rust CLI nor Studio can host these — the sidecar ships its own
console face instead. `caliper-learn` exits **1 when any error-severity finding
was reported**, else 0, so CI can gate without parsing output (same hook as
`caliper report --strict`). Every subcommand takes `--json`.

---

## Eval harness (`E001`–`E003`)

Training loss predicts almost nothing about closed-loop competence — BC
covariate shift, chunk-cadence mismatches, and normalization drift all hide
behind a pretty loss curve. The only honest metric is rollouts:
`evaluate(policy, task, cfg)` runs N seeded episodes on `VecSimEnv` and
reports what actually happened.

**Aggregate semantics.** Episode *k* runs on seed `base_seed + k` (seeded
init jitter around the joint-range midpoints); **success** is defined as the
task's `termination_fn` firing within `max_steps`. Each episode row carries
its `seed`, `success`, `steps`, summed `episode_return`, and — when the task
defines a `distance_fn` (`reach_eval_task` does) — the `final_distance`, so a
failed episode still says whether it was *almost* or *nowhere near*. The
success rate is aggregated with a **Wilson 95% score interval**, chosen over
the normal approximation because eval runs are small and rates sit at the
edges: 0/N and N/N still get honest, non-degenerate intervals, and a 3/5
result is reported as the coin-flip it is (CI ≈ [0.23, 0.88]) instead of
"60%". Deterministic end to end: the same `EvalConfig` and a deterministic
policy produce a byte-identical `to_json(result)`; a policy with its own
unseeded RNG breaks that — seed it in `reset()`, the way the diffusion head
does.

`sweep(checkpoints, task, cfg)` is the checkpoint-selection answer: every
candidate — Hub checkpoint directories *and* in-memory policies or scripted
callables, in one table — is evaluated under the **same seeds** and ranked by
success rate, then mean return, then name (a stable total order).

**E001 — all episodes failed** *(Warning)*. 0/N episodes reached termination.
Why it matters: this is the "loss went down but the policy does nothing"
headline, made unmissable. Fix: check the deploy cadence (eval fps vs
collection fps — action chunks consumed at the wrong rate), the observation
feature mapping (right robot, right dof count), and `sweep()` the other
checkpoints before blaming the data. If it persists, run the
[autopsy](#the-autopsy) — the cause is usually upstream.

**E002 — seed lottery** *(Warning)*. Success flips seed-to-seed:
`0 < successes < N` **and** the Wilson interval spans more than 0.5. Why it
matters: a run this noisy cannot distinguish checkpoints — picking one on it
is picking on luck (2/4 fires at width 0.70; 50/100 stays silent at
width < 0.5). Fix: raise `EvalConfig.n_episodes`; the interval shrinks
~1/√n.

**E003 — zero reward signal** *(Warning)*. Every episode returned exactly
0.0. Why it matters: only a missing `reward_fn` produces this, and success
alone cannot rank near-misses — returns stay uninformative and E002-style
noise cannot be diagnosed away. Fix: wire a `reward_fn` into `EvalTask`
(e.g. `reach_eval_task`).

```python
from caliper_learn import EvalConfig, evaluate, reach_eval_task, sweep

task = reach_eval_task(robot, "tool0", [0.4, 0.0, 0.3], tol=0.05, fps=50)
result = evaluate(policy, task, EvalConfig(n_episodes=20, base_seed=0))
ranking = sweep({"ckpt_3k": "runs/003000/pretrained_model", "baseline": my_fn}, task)
```

## Latency profiler (`L001`–`L003`)

`profile_rollout(policy, control_loop, ticks=200, fps=50)` drives the policy
through the same three stages as the deploy runner — obs build → inference →
`step_with_target` — timing each per tick with `perf_counter_ns`. The
headline is deliberately pessimistic: **`achievable_hz = 1 / p95(tick
total)`**, the rate the loop holds on 95% of ticks, not the average that
hides the spikes. The profiler's own scaffold cost is measured on an empty
loop first and subtracted, so the report charges the policy and the engine,
not the instrumentation.

The split that matters is **chunk-aware**: lerobot-style `select_action` pops
an internal queue and only re-runs the network every `n_action_steps` ticks,
so *mean* inference time is a lie — the **refill tick** is the one that must
fit the budget. Refill ticks are identified from the policy's chunk config
when available (else detected from timing bimodality, with an absolute floor
so scheduler noise on a microsecond-fast policy is never misread as
chunking), and refill p95 is reported separately from pop p95.

**L001 — budget exceeded** *(Error)*. More than 5% of ticks exceeded the
`1/fps` budget — the p95 tick does not fit, and the requested rate is
dishonest. Why it matters: on hardware the loop either slips (cadence
mismatch — the exact failure class the debugger's P005 catches at the config
level) or back-pressures the controller. Fix: run at ≤ the reported
achievable Hz *and collect/retrain at that fps* — train and deploy must share
one cadence — or cut the dominant stage in the table.

**L002 — inference dominates** *(Info)*. Inference is over 60% of the median
tick *and* the tick is a material fraction (>20%) of the budget — a 5 µs
loop that is "90% inference" stays silent. Why it matters: it tells you where
optimization effort pays; obs build and the engine step are not the
bottleneck. Fix: shrink or `torch.compile` the model, or raise
`n_action_steps` so the forward pass amortizes — while watching the refill
p95, because that single tick still has to fit the budget.

**L003 — high jitter** *(Warning)*. Tick-period std beyond
max(25% of budget, 1 ms): the cadence is unstable even if the average holds.
Why it matters: chunked policies assume evenly-spaced actions; jittered
delivery deforms every executed trajectory. Fix: look for periodic spikes
first (chunk refills — the refill/pop split shows them), then background
load and GC pauses; pin the process or lower the fps until the period
stabilizes.

```python
import caliper
from caliper_learn import profile_rollout

loop = caliper.ControlLoop(robot, dt=1 / 50, start=[0.0] * robot.ndof)
report = profile_rollout(policy, loop, ticks=200, fps=50)
print(report.render_text())   # per-stage p50/p95/p99/max + refill-vs-pop split
```

## Policy debugger (`P001`–`P008`)

`analyze_policy(policy_dir, dataset_root=None, robot=None)` is the deploy
debugger: checkpoint in, "why does my trained policy do nothing" out. It
inspects a lerobot-Hub-convention checkpoint (via the safetensors-only `hub`
loader — same security stance as deploy), probes its forward pass on
dataset-replayed observations, and names the mined failure modes.
`dataset_root` unlocks P002/P004/P005 and dataset-replayed probes; `robot`
unlocks P003. Static config checks run **first**, so a checkpoint lerobot's
own parser would crash on still gets a calm diagnosis instead of a stack
trace. Behavioral thresholds are empirically calibrated (a random-init policy
measures ≥ 0.17 normalized action spread and ≥ 0.19 dead-input response;
collapsed weights measure exactly 0.0 — the 0.05/0.02 cuts sit mid-gap), and
every probe forward is preceded by `policy.reset()` so the chunk queue never
serves a stale action.

**P001 — action collapse** *(Error)*. Every probe state returns (nearly) the
same action. Why it predicts deploy failure: the policy is a constant —
typically the dataset mean, which is the L2 optimum for unlearnable labels,
so **training loss looks fine**. Fix: check for zeroed/corrupted weights,
then whether the state→action map is one-to-many (the sidecar's
one-step-lookahead labeling exists precisely to fix that; see also the
dataset doctor's D006).

**P002 — per-dof collapse** *(Warning, needs `dataset_root`)*. A dof the
*data* moves but the policy never does — the network wrote that joint off.
Why: at deploy the joint simply does not track, and nothing errors. Fix:
look at the dof's loss contribution and its normalization std — a wrongly
large per-dof std makes its normalized targets vanish, and the network learns
to ignore the joint.

**P003 — joint-limit saturation** *(Warning, needs `robot`)*. Actions land
outside a joint's URDF limits on >20% of probes. Why: the `SafetyMonitor`
clamps every tick, so the executed motion is not what the policy "intended" —
it runs, but wrong. Fix: almost always an unnormalization-scale problem
(check P004 and the action std in the postprocessor stats); genuine
limit-riding demonstrations are rare.

**P004 — normalization mismatch** *(Error, needs `dataset_root`)*. The
processor's train-time stats (read straight from the checkpoint's
safetensors — no model load needed) disagree with stats recomputed from the
dataset. Why: this is the killer — every input and output is silently
shifted and scaled, the policy trains in one coordinate system and deploys in
another. It is the checkpoint-side twin of the dataset doctor's D002. Fix:
the classic causes are a dataset edited/regrown after training or the wrong
checkpoint paired with this dataset — retrain, or regenerate the processor
stats from *this* dataset.

**P005 — cadence mismatch** *(Error, needs `dataset_root` + a
`train_config.json` that declares an fps)*. The checkpoint's recorded
training fps disagrees with the dataset fps. Why: action chunks are consumed
at the wrong rate — each queued action is "worth" a different amount of real
time than the one it was trained to be (collecting at 50 Hz and deploying at
1 kHz once closed only ~42% of the gap in this repo). Fix: deploy at
`dt = 1/fps` of the collection cadence; train and deploy must share one fps.

**P006 — dead input** *(Warning)*. Perturbing one state dimension (±0.25 of
its data std) never changes the action across the probe bases. Why: the
policy cannot close the loop on that joint's measurement — if the task needs
it, it will fail open-loop-style. Fix: check the feature wiring, and whether
the dim was constant in training (the dataset doctor's D001 — in which case
the dataset, not the policy, is the defect). Image-input probes are honestly
`NotImplemented` this wave; the loader gates VISUAL checkpoints anyway.

**P007 — non-finite forward** *(Error)*. NaN/inf anywhere in the probed
actions. The other behavioral checks are **skipped** — their math would be
garbage-on-garbage. Fix: inspect `model.safetensors` for NaN/inf tensors and
the training run for loss spikes.

**P008 — chunk-config anomaly** *(Error or Info)*. `n_action_steps >
chunk_size` (the queue would pop more steps than a forward pass produces) or
temporal ensembling with `n_action_steps != 1` are Errors — lerobot's own
parser crashes on these, so the debugger diagnoses them **statically from
`config.json` before any model load** and skips the load entirely.
`chunk_size > 1` with `n_action_steps = 1` and no ensembling is an Info: the
network re-runs every tick and throws away all but one predicted action —
legal, just wasteful (the profiler's L002 will usually confirm).

```python
from caliper_learn import analyze_policy, render_policy_findings

findings = analyze_policy("runs/003000/pretrained_model",
                          dataset_root="data/reach_demos", robot=robot)
print(render_policy_findings(findings))   # [] means every reachable check passed
```

## The autopsy

`autopsy(policy_dir, dataset_root, robot=None, task=None)` merges every
diagnostic that applies into one `AutopsyReport` with one verdict:

- **D-section** — `caliper.data_doctor` on the dataset ([`D001`–`D015`](doctors.md#dataset-doctor-d001d015); v3.0
  on-disk format — the doctor's own error names the converter for v2.x).
- **P-section** — `analyze_policy`, dataset-aware.
- **E-section** — `evaluate`, Wilson-95. Only when `robot` **and** `task`
  are given (rollouts need a sim).
- **L-section** — `profile_rollout` on a fresh `ControlLoop`. Same gating.

The **verdict** paragraph is template-based and honest: it leads with the
most severe section, and ties break toward the dataset — data problems cause
policy problems, not the other way around, so the upstream fix comes first.

### Walkthrough

A policy trained on a teleop dataset "converged" (loss looked fine) but does
nothing useful in sim. One command:

```console
$ caliper-learn autopsy runs/reach_act/pretrained_model data/reach_demos \
    --urdf arm.urdf --frame tool0 --target 0.40 0.00 0.30 --episodes 20
```

The report (episode rows and long messages trimmed for width):

```text
== Caliper autopsy ==
policy:  runs/reach_act/pretrained_model
dataset: data/reach_demos

VERDICT: The dataset has 0 error(s) and 4 warning(s) (D001, D004, D009, D011)
that predict training failure; the policy checks are clean; closed-loop: 0/20
episodes succeeded (95% CI [0.00, 0.16]) — the policy never solved the task;
the deploy loop holds 50 Hz with headroom.

-- dataset doctor (D) — 40 episodes, 8000 frames @ 50 fps --
  [D001] (warning) feature=observation.state dof=5 feature 'observation.state'
         dof 5 ('wrist_roll'): constant at 0.000000 across all 8000 frames — …
  [D004] (warning) feature=action 'action' is nearly identical to
         'observation.state' (rms difference 0.000731 vs state spread 0.4120) —
         echo/lag labels; the policy can minimize loss by copying its input …
  [D009] (info) episode=17 episode 17: length 2311 frames vs a median of 190 …
  [D011] (warning) episode=31 episode 31: the last 74 frames are bit-identical …

-- policy debugger (P) --
no findings — every reachable policy check passed.

-- closed-loop eval (E) --
episodes: 0/20 succeeded  success_rate=0.000  wilson95=[0.000, 0.161]
return: mean=-121.0483 median=-118.5210  steps-to-success: mean=-
    seed  success  steps       return  final_dist
       0       no    200    -104.5121      0.4818
       1       no    200    -131.2246      0.6103
       …
[WARN] E001: 0/20 episodes reached termination — the policy never solved the task.
    fix: Training loss says nothing about this. Check the deploy cadence …

-- deploy latency (L) --
Latency profile — 100 ticks @ 50 Hz (budget 20.000 ms/tick)
  achievable: ~152 Hz (1 / p95 tick time); 0.0% of ticks over budget
  …
  chunk queue (config): refills every 8 ticks (13 seen) — refill p95 6.104 ms
  vs pop p95 0.058 ms
no findings — the loop holds 50 Hz with headroom.
```

**How to read the verdict.** Read it left to right — it is ordered by blame:

1. *"The dataset has … 4 warning(s) … that predict training failure"* leads,
   because the dataset section is the most severe and ties go upstream. D004
   is the actual killer here: the action labels echo the state, so the loss
   was minimized by copying the input — the policy honestly learned exactly
   what the data taught.
2. *"the policy checks are clean"* — and note what the autopsy did **not**
   do: the data-dead `wrist_roll` (D001) is correctly *not* blamed on the
   policy — P002 only judges dofs the data moves. A clean P-section plus a
   defective D-section says: don't debug the checkpoint, fix the data.
3. *"closed-loop: 0/20 …"* quantifies the damage with an honest interval
   ([0.00, 0.16] — even the best case is bad), and
4. *"the deploy loop holds 50 Hz with headroom"* rules out the remaining
   suspect: this is not a latency problem.

The fix, per the finding hints: retrain on one-step-shifted (or delta)
action labels, trim episode 31's frozen tail and review episode 17 with the
dataset edit ops, and decide whether `wrist_roll` should move. Then run the
same command again — the report is the regression test.

Programmatic use mirrors the CLI one-to-one:

```python
from caliper_learn import autopsy, reach_eval_task

rep = autopsy("runs/reach_act/pretrained_model", "data/reach_demos",
              robot=robot, task=reach_eval_task(robot, "tool0", [0.4, 0.0, 0.3]))
print(rep.verdict)
rep.to_json(indent=2)   # sorted keys; D/P/E sections byte-deterministic
```

## Honest scope

- **State-only, this wave.** Eval observations and debugger probes cover
  state features; image observations (and P006 image-input probes) arrive
  with the vision wave — the `hub` loader gates VISUAL checkpoints with a
  clear `NotImplementedError` today, so nothing fails silently.
- **No Studio panel, by design.** Policy inference happens Python-side
  (torch), and Studio's backend is the Rust engine — there is no autopsy
  button in the app. The terminal face is `caliper-learn`; its `--json`
  output is the integration point if a panel ever wants to render it.
- **The L-section is wall-clock** and therefore honestly non-deterministic;
  everything else in an autopsy (D/P/E) serializes byte-identically for the
  same inputs.
- **Environment**: eval needs `mujoco` (via `VecSimEnv`), the profiler and
  autopsy L-section need a `caliper.ControlLoop` (the robot needs inertial
  data — run the [asset doctor](doctors.md) if `has_inertia` is false), and
  loading Hub checkpoints needs `torch` + `lerobot`. All of it imports
  lazily; `caliper-learn --help` is instant.
