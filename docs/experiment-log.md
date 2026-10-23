# Experiment log

## Data collection and preparation

The initial collection contained 22 demonstrations. Original episode 13 was
excluded because the cube was not released, leaving 21 episodes with 11,894
observations. XYZ delta-action computation produced 11,873 transitions.
Raw recordings, processed datasets, and checkpoints are local and ignored by
Git. The keyboard video in `media/` is a teleoperation reference, not a learned
policy rollout.

## Exercise 1: behavioral cloning baseline

- Observation: EE XYZ, gripper angle, cube XYZ, obstacle XYZ (10 values).
- Action: EE XYZ deltas and absolute gripper command (4 values).
- Architecture: two 128-unit ReLU hidden layers, 16-action output chunk;
  26,176 parameters.
- Loss: MSE on normalized action chunks.
- Optimizer: Adam, learning rate 0.001; cosine scheduler to 0.00001 over 100 epochs.
- Batch size: 64; training seed: 42.
- Split: 19 training episodes and 2 validation episodes. Validation IDs were
  `[15, 8]`, zero-based in the curated dataset; overlapping chunks never cross
  the split. Normalization used training rows only.
- Training chunks: 10,502; validation chunks: 1,056.
- Best checkpoint: epoch 8, validation loss 0.651812.
- Preserved file: `checkpoints/single_cube/baseline_21ep_ee_xyz_obstacle.pt`.
- Evaluation: 61 successes out of 100 rollouts, seed 123, default obstacle
  distribution, default 800-step limit.

Visual observations included initial trembling, a pause before opening, and
failed releases onto the bin wall. The release diagnostic measures the first
opening command near the bin, not the exact moment the cube leaves the fingers.
Across the original 21 episodes, the mean Y offset from bin center was -2.33 cm.

## Additional five-episode attempt

Five demonstrations were initially added, giving 26 episodes total with the
same original validation IDs. Their mean opening-position Y offset was -1.45 cm.
The best checkpoint was epoch 2, with validation loss 0.650832. Evaluation
reported 0/100 successes; the policy reached the bin but failed to release.
Those five raw recordings and their separate processed dataset were later
deleted. The original baseline backup was preserved.

## Seven replacement demonstrations

Session: `2026-10-11_15-20-46`, seven episodes, 3,721 raw observations and 3,714
processed transitions in `datasets/processed/single_cube/new_7_ee_xyz.zarr`.
Mean opening-position offsets: X -1.30 cm, Y -1.69 cm.

All seven recordings contained opening commands near the bin, final cube
positions inside the bin, and 17–37 recorded steps after the last opening
command. This supplies enough ending context for a 16-action chunk and does
not support early termination as the cause of the new release failure.

### Combined collection: 28 episodes

The original dataset was loaded first and the seven new episodes appended.
Validation remained on original episode IDs `[15, 8]`: 26 training episodes and
2 validation episodes. The best checkpoint was epoch 4, validation loss
0.640663. Rollouts again reached the bin but did not release; success was
reported as 0%. A separate retained 100-rollout log for this experiment is not
available, so its exact denominator is not asserted here.

A possible explanation is that differing approach and release strategies
produce ambiguous targets at similar observations. MSE can average those
commands. This is a hypothesis, not a verified causal diagnosis; model fit,
checkpoint selection, and rollout distribution remain possible contributors.

### Seven-only diagnostic

Six episodes were used for training and one for validation with seed 42.
The best checkpoint was epoch 24, validation loss 0.519916, preserved as
`checkpoints/single_cube/new7_only_ee_xyz_obstacle.pt`.
Visual evaluation used seed 123 and three requested episodes. The policy failed
around the obstacle before reaching the bin. No 100-rollout success rate was
measured, and this experiment could not isolate release behavior.

Normalized validation losses across datasets are not directly comparable
because training-derived normalization and validation splits can differ.
Lower validation error alone does not establish improved task success.

The seven-only run overwrote `best_model_ee_xyz_obstacle.pt`; the combined
28-episode checkpoint was not preserved separately. The original 21-episode
baseline remains intact and was visually confirmed to release successfully.

## Exercise 2: DAgger

### Changed obstacle distribution

Both obstacle sections are stationary during a rollout and move together along
X at reset. Adversarial placement samples the center zone with probability 20%,
and side zones displaced by +8 cm and -8 cm with probability 40% each. The
center-zone standard deviation is 1 cm; side-zone standard deviation is 0.5 cm.
The initial baseline audit, seed 123, failed all three inspected adversarial
rollouts before lifting the cube. This is a three-rollout observation, not a
100-rollout adversarial baseline estimate.

Successful manual routes were tested for the first two seeded side placements.
The negative-X placement required a different approach: move toward positive X,
lower slightly before aligning with the cube, then lift in two stages around a
sideways movement. The refined routes completed pickup and release with no
obstacle contact detected at the sampled control steps. These were scripted
key-handler tests, not learned-policy successes or a universal route guarantee.

### First correction round: 10 curated episodes

Eleven corrections were initially saved. Session `2026-10-11_21-10-39`, episode 2
(the third correction overall), was excluded following the learner's assessment
that it was unsuccessful. A filtered copy preserves the source. The curated
collection contains 10 episodes and 3,725 processed transitions.

Training retained the 21 original demonstrations, held out original episode IDs
`[15, 8]`, and placed all corrections in training: 29 training episodes and 2
validation episodes. Architecture and optimizer settings initially remained
unchanged at width 128, depth 2, chunk size 16, seed 42, and 100 epochs.

The best-validation checkpoint was epoch 3 with loss 0.611136. In three
diagnostic adversarial rollouts, it failed to grasp in one and held the cube
near the bin without releasing in two. Its gripper commands near the bin stayed
below 0.29 rad and were applied correctly. In a separate diagnostic intervention,
holding the end-effector target fixed and applying a 0.6-rad opening command
released the cube into the bin after two steps. This intervention is not counted
as policy success.

Nearly identical observations within one correction had expert gripper commands
of 0.026 and 0.326 rad, supporting the presence of ambiguous hold/open targets.
MSE averaging remains a possible contributor, not a complete causal diagnosis.
The early checkpoint also fitted demonstrated opening actions poorly.

A final-epoch checkpoint was added alongside the best-validation checkpoint.
Epoch 100, validation loss 0.641567, moved more smoothly and opened over the bin.
The learner reported 14/100 adversarial successes using seed 123. Original-task
performance for this checkpoint was not reported.

### Second correction round: 24 curated episodes

Added 14 corrections from sessions `2026-10-11_21-41-01` (10 episodes) and
`2026-10-11_21-50-11` (4 episodes). All recorded final cube positions passed an
in-bin position check. The full curated correction dataset contains 24 episodes,
7,510 observations and 7,486 processed transitions. It includes the earlier 10
corrections once; the excluded failed episode remains excluded.

Combined training used 45 episodes and 19,359 transitions: 43 training episodes,
2 original validation episodes, 17,628 training chunks and 1,056 validation
chunks. All other training settings remained unchanged. The 128-unit model's
best checkpoint was epoch 5, loss 0.574608; epoch 100 had loss 0.610123.

### Capacity experiment: width 512

Hidden width increased from 128 to 512, keeping two hidden layers, data, split,
chunk size, optimizer, schedule, epochs, and seed unchanged. Parameter count
increased from 26,176 to 301,120. Width 256 was discussed but not tested.

Training now saves `d_model` and `depth` in the checkpoint so the evaluation
loader reconstructs the network correctly. Width-specific filenames preserve
previous experiments. The best checkpoint was epoch 3 with loss 0.579185;
epoch 100 had loss 0.630460.

### Reported rollout results

The learner reported the following results from 100-rollout evaluations with
seed 123 and an 800-step limit. Original and adversarial distributions are
reported separately. These numbers were provided in the conversation; complete
saved evaluation logs were not supplied for every historical run.

| Corrections | Hidden width | Checkpoint | Original success | Adversarial success |
|---:|---:|---|---:|---:|
| 0 | 128 | Original baseline, epoch 8 | 61/100 | Not measured over 100 |
| 10 | 128 | Final, epoch 100 | Not reported | 14/100 |
| 24 | 128 | Best validation, epoch 5 | 11/100 | 8/100 |
| 24 | 128 | Final, epoch 100 | 19/100 | 45/100 |
| 24 | 512 | Best validation, epoch 3 | 90/100 | 12/100 |
| 24 | 512 | Final, epoch 100 | 9/100 | 38/100 |

The width-512 early checkpoint improved original-distribution success compared
with the initial baseline, while the final checkpoint handled more adversarial
rollouts. Neither checkpoint demonstrated high success on both distributions.
These results show a performance tradeoff, not proof that MSE alone causes it.
The validation set contains only original demonstrations, so its loss does not
measure performance on held-out adversarial corrections.

### Width-512 checkpoint comparison: completed screening

A separate `checkpoints/single_cube/epoch_comparison/` directory preserves the
previous checkpoints. With the same width-512 setup, training saved checkpoints
every 10 epochs, plus the best-validation and final models. All stored epoch
numbers and validation losses were checked against the pasted training log.

| Epoch | Original-demonstration validation loss |
|---:|---:|
| 3 (best) | 0.579185 |
| 10 | 0.601016 |
| 20 | 0.610040 |
| 30 | 0.622489 |
| 40 | 0.624015 |
| 50 | 0.620847 |
| 60 | 0.637956 |
| 70 | 0.634861 |
| 80 | 0.628474 |
| 90 | 0.631708 |
| 100 | 0.630460 |

Evaluated the best-validation checkpoint and all ten periodic checkpoints on
both obstacle distributions with seed 789, 50 rollouts per checkpoint per
distribution, and an 800-step limit. All 22 saved logs were checked against the
reported summaries. Full local logs are in `logs/epoch_comparison/` and are
ignored by Git.

| Epoch | Original success | Adversarial success | Lower of the two rates |
|---:|---:|---:|---:|
| 3 (best validation) | 43/50 (86%) | 14/50 (28%) | 28% |
| 10 | 27/50 (54%) | 32/50 (64%) | 54% |
| 20 | 0/50 (0%) | 13/50 (26%) | 0% |
| 30 | 0/50 (0%) | 25/50 (50%) | 0% |
| 40 | 14/50 (28%) | 12/50 (24%) | 24% |
| 50 | 0/50 (0%) | 27/50 (54%) | 0% |
| 60 | 2/50 (4%) | 18/50 (36%) | 4% |
| 70 | 17/50 (34%) | 20/50 (40%) | 34% |
| 80 | 13/50 (26%) | 18/50 (36%) | 26% |
| 90 | 8/50 (16%) | 12/50 (24%) | 16% |
| 100 | 10/50 (20%) | 15/50 (30%) | 20% |

Epoch 10 is the strongest balanced candidate in this screening: original success
54% and adversarial success 64%, with a minimum of 54%. It also has the highest
adversarial success among the tested checkpoints. The best-validation model
retains the highest original success (86%) but reaches only 28% adversarial
success. Later epochs do not improve either distribution monotonically.

These are checkpoint-selection measurements on 50 episodes, not an independent
final estimate. Results on seed 789 should not be directly substituted for the
earlier 100-rollout measurements on seed 123. Confirm the eventual selected
model on a separate 100-rollout seed after completing the width comparison.

### Width-256 comparison: completed screening

Repeated the width comparison with hidden width 256 (84,992 parameters), keeping
both hidden layers, data, original validation IDs `[15, 8]`, chunk size 16,
training seed 42, optimizer, schedule and 100 epochs unchanged. Evaluated all
periodic checkpoints and the best-validation model on 50 episodes per obstacle
distribution with seed 789 and an 800-step limit. All 22 saved logs under
`logs/epoch_comparison_width256/` were verified against the reported results.

| Checkpoint | Original success | Adversarial success |
|---|---:|---:|
| Best validation | 0/50 (0%) | 0/50 (0%) |
| Epoch 10 | 48/50 (96%) | 45/50 (90%) |
| Epoch 20 | 34/50 (68%) | 42/50 (84%) |
| Epoch 30 | 1/50 (2%) | 31/50 (62%) |
| Epoch 40 | 1/50 (2%) | 28/50 (56%) |
| Epoch 50 | 5/50 (10%) | 23/50 (46%) |
| Epoch 60 | 7/50 (14%) | 29/50 (58%) |
| Epoch 70 | 14/50 (28%) | 27/50 (54%) |
| Epoch 80 | 15/50 (30%) | 24/50 (48%) |
| Epoch 90 | 10/50 (20%) | 29/50 (58%) |
| Epoch 100 | 15/50 (30%) | 27/50 (54%) |

Width 256, epoch 10 is the strongest candidate across both width comparisons:
96% original and 90% adversarial success, compared with width 512, epoch 10 at
54% and 64%. These are selection measurements on the same evaluation seed, not
independent final-test estimates or evidence that width 256 always outperforms
width 512 across training seeds.

Comparing the strongest balanced checkpoint from each run, width 256 epoch 10
achieved a lower-distribution success rate of 90%, while width 512 epoch 10
achieved 54%. The larger 512-unit model therefore performed worse by this
selection criterion in these experiments. Success varied sharply between saved
epochs for both widths; additional training was not a reliable improvement.
The best-validation rows are not epoch-0 measurements. No untrained epoch-0
policy was evaluated. The README includes the full side-by-side percentage
comparison; the tables above retain exact counts.

Selected checkpoint:
`checkpoints/single_cube/epoch_comparison/epoch010_ee_xyz_obstacle_dagger24ep_width256.pt`.
Later checkpoints' lower success does not invalidate the saved epoch-10 weights.
No retraining is required merely because later epochs performed poorly.

### Selected model: separate-seed confirmation

Evaluated the selected width-256 epoch-10 checkpoint using seed 987, 100 episodes
per distribution, an 800-step limit and execution horizon 16. Both evaluation
processes completed successfully:

| Distribution | Successes | Rate |
|---|---:|---:|
| Original | 97/100 | 97% |
| Adversarial | 82/100 | 82% |

The selected checkpoint stored epoch 10 and validation loss 0.593918. These
confirmation runs used CPU inference because CUDA was unavailable in the tool
execution environment, whereas the earlier learner-run evaluations used CUDA.
Headless rendering used EGL; CPU inference used one OpenMP/MKL thread. The
initial attempts were interrupted without producing results, then rerun from
seed 987 with unbuffered output. The completed runs used the original evaluation
script and did not change policy weights or simulation settings.

Full logs and a machine-readable summary are saved locally under
`logs/confirmation_width256_seed987/` (ignored by Git). The 82% adversarial
confirmation result replaces 90% screening success as the fresh-seed estimate;
retain both results with their sample counts and seeds.

### Execution-horizon experiment

Kept the selected width-256 epoch-10 checkpoint and prediction chunk size 16,
but executed only the first 8 actions before replanning from a new observation.
The evaluation option `--execution-horizon` changes the action queue without
changing policy weights, normalization or success criteria. Omitting the option
executes the full predicted chunk.

Learner-run CUDA evaluations with seed 987 produced zero successes in the
observed trials on both obstacle distributions. The supplied log shows 0/87
completed original episodes before interruption and 0/28 completed adversarial
episodes at the end of the excerpt. These are partial observations, not two
completed 100-episode evaluations.

A controlled CPU diagnostic using the updated script, identical checkpoint,
seed 987 and three episodes per distribution reproduced the difference:

| Execution horizon | Original | Adversarial |
|---:|---:|---:|
| 16 | 3/3 | 2/3 |
| 8 | 0/3 | 0/3 |

The learner also confirmed that restoring horizon 16 restored normal behavior.
Retain horizon 16 for this checkpoint. Replanning sooner is not automatically
better: restarting prediction after 8 steps discards the latter half of each
chunk and changes the resulting control sequence. The exact physical failure
mechanism was not established. The three-episode diagnostic is not a new
100-episode performance estimate; the reference results remain 97/100 original
and 82/100 adversarial at horizon 16.

### Further experiments discussed, not yet run

- Balance sampling across original and both shifted obstacle zones.
- Reduce chunk size from 16 to 8.
- Test other execution horizons, such as 4 or 12; horizon 8 failed above.
- Reduce initial learning rate from 0.001 to 0.0003.
- Try AdamW with weight decay 0.0001.
- Compare network depth or add joint angles to observations.
- Add held-out adversarial episodes and report validation losses by distribution.

These are proposed settings, not established improvements. Change one setting
at a time and record both rollout success rates and the checkpoint-selection
procedure.

### Successful policy videos

Recorded two autonomous rollouts on 2026-10-12 with the selected width-256,
epoch-10 checkpoint, seed 987, CPU inference and execution horizon 16. The first
episode succeeded in both distributions: 592 control steps for the original
obstacle, 417 for the adversarial obstacle. The standard `check_success` criterion
was used. Source frames were captured at 10 Hz, matching the control loop,
with a 2-second final-frame hold. Published videos play at **3× simulation
speed**, with an on-screen label and 30 frames per second to retain the source
frames. Their durations are approximately 20.47 seconds (original) and 14.63
seconds (adversarial), including the sped-up final hold. No human actions
were applied. These illustrative successes do not change aggregate evaluation
results. MP4 videos, preview images and metadata are tracked under `media/` and
linked from the README.

## Exercise 3: multicube baseline preparation

After removing four incorrectly labeled episodes from the initial session,
the retained raw multicube collection contains 42 episodes: 11 red, 15 green,
and 16 blue. One empty session was skipped. Preprocessing produced 10,400
EE-XYZ transitions. Stored one-hot labels were verified to be valid and constant
within each episode; this does not independently verify which cube was moved.

Implemented `MultiTaskPolicy` as a two-hidden-layer width-256 ReLU MLP with
19 observation inputs, 16 predicted actions and 4 values per action (87,360
parameters). Inputs are EE XYZ, gripper opening, the current XYZ positions of
all three cubes, target-color one-hot and bin XYZ. Loss remains normalized MSE.
The factory now forwards chunk size and architecture settings to this policy.
Multitask training defaults include all these input keys and both action keys.

A color-stratified episode split with seed 42 reserves zero-based episodes
[36, 9, 32] for validation: one red, one green and one blue. Training uses
39 episodes (10 red, 14 green, 15 blue), with 9,031 valid chunks; validation
has 739 chunks. Three validation episodes are a small diagnostic holdout,
so rollout evaluation remains necessary. The single-cube fixed split is retained.

Prediction shape, finite loss and gradients passed checks. A temporary one-epoch
CUDA training run completed and its checkpoint reloaded with the correct model
and input metadata. Outputs for this initial check were written only under /tmp. The full run is
reported below; no multicube success rate is available yet.

### First full multicube training run

Completed 100 epochs on CUDA with training seed 42, width 256, depth 2,
chunk size 16, batch size 64, Adam learning rate 0.001 and the existing cosine
schedule. The dataset and per-color holdouts are as described above.

Best validation MSE was 0.887824 at epoch 2. At epoch 100, training MSE was
0.593785 and validation MSE was 1.095338. Training error declined while
validation error generally rose, consistent with overfitting this small
held-out set. This alone does not establish rollout success or its best epoch.

Saved best-validation, every-10-epoch and final checkpoints under
`checkpoints/multi_cube/epoch_comparison/`, with the suffix
`ee_xyz_multitask_width256.pt`. The complete log is
`logs/multicube/train_width256_seed42.log`. These artifacts are ignored by Git.
Next, evaluate saved checkpoints with shuffled multicube layouts and report
success separately for each target color before selecting a model.

### Multicube checkpoint screening

Evaluated the best-validation and all ten periodic checkpoints with seed 789,
30 episodes per checkpoint (10 per target color), shuffled layouts, an 800-step
limit and execution horizon 16. Full logs and results.json are saved locally
under `logs/multicube/checkpoint_comparison/`.

| Checkpoint | Overall | Red | Green | Blue |
|---|---:|---:|---:|---:|
| Best validation (epoch 2) | 0/30 | 0/10 | 0/10 | 0/10 |
| Epoch 10 | 1/30 | 0/10 | 0/10 | 1/10 |
| Epoch 20 | 1/30 | 0/10 | 0/10 | 1/10 |
| Epoch 30 | 3/30 | 0/10 | 0/10 | 3/10 |
| Epoch 40 | 1/30 | 0/10 | 0/10 | 1/10 |
| Epoch 50 | 0/30 | 0/10 | 0/10 | 0/10 |
| Epoch 60 | 1/30 | 0/10 | 0/10 | 1/10 |
| Epoch 70 | 0/30 | 0/10 | 0/10 | 0/10 |
| Epoch 80 | 1/30 | 0/10 | 0/10 | 1/10 |
| Epoch 90 | 0/30 | 0/10 | 0/10 | 0/10 |
| Epoch 100 | 0/30 | 0/10 | 0/10 | 0/10 |

Epoch 30 achieved the highest observed overall success, 3/30 (10%), all on
blue-target episodes. No tested checkpoint succeeded on red or green in these
ten episodes per color. These are screening results, not independent final
estimates; the small sample does not prove that other colors can never succeed.
The plain concatenation MLP is not yet a strong multitask policy.

A separate three-episode CPU diagnostic of epoch 30 with seed 789 showed the
red target lifted to about 9.2 cm but still outside the bin at step 800; the
green target remained at table height; the blue target reached the bin at step
206. This confirms different failure stages in those examples, not a general
causal explanation. Diagnostic details are retained locally in
`logs/multicube/checkpoint_comparison/epoch30_diagnostics.json`.

Next investigate demonstration target correctness and layout coverage, then
consider explicitly selecting the target cube position and expressing movement
relative to the robot and bin. Representation changes require consistent training
and inference processing and a retrained model; do not alter the existing weights.

### Width/depth comparison: best-validation checkpoints only

Added `--d-model`, `--depth` and `--checkpoint-dir` training options. Ran three
additional 100-epoch experiments, preserving seed 42, all data and validation
IDs, chunk size 16, batch size 64, Adam and learning-rate schedule. Outputs
were saved in separate directories to preserve the original baseline.
Only each run's lowest-validation-MSE checkpoint was evaluated, as requested;
periodic checkpoints were saved but not evaluated. Evaluation used seed 789,
30 episodes (10 per color), shuffled layouts, 800 steps and execution horizon 16.

| Width | Hidden layers | Parameters | Selected epoch | Validation MSE | Overall | Red | Green | Blue |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 (baseline) | 2 | 87,360 | 2 | 0.887824 | 0/30 | 0/10 | 0/10 | 0/10 |
| 128 | 2 | 27,328 | 2 | 0.877187 | 1/30 | 0/10 | 1/10 | 0/10 |
| 256 | 3 | 153,152 | 1 | 0.883566 | 0/30 | 0/10 | 0/10 | 0/10 |
| 512 | 2 | 305,728 | 1 | 0.882735 | 0/30 | 0/10 | 0/10 | 0/10 |

The width-128 model had one green-target success; the other alternatives had
none. This small screening provides no clear task-performance improvement from
changing width or depth under best-validation checkpoint selection. It does not
rule out successful later checkpoints or other training seeds. The earlier
baseline epoch-30 result (3/30) is a separate rollout-selected checkpoint and
should not be treated as the baseline best-validation result in this comparison.

Train/evaluation logs and results.json are stored locally under
`logs/multicube/capacity_comparison/`. Checkpoints are under
`checkpoints/multi_cube/capacity_comparison/width*_depth*/`.
Next examine target correctness/layout coverage and test an explicit target
position or relative-coordinate representation as a separate experiment.

### Goal-relative feature experiment: implementation ready

The learner's demonstration check found the correctly labeled target cube in
the bin at the final recorded step of all 42 retained episodes. This supports
final target-label consistency; trajectory quality and layout coverage were
not established by that check.

Added a shared `goal_relative_features` helper and the virtual state key
`goal_relative`. Dataset loading and live observation assembly both select the
target XYZ using the one-hot goal, then compute target-minus-EE and bin-minus-EE
before normalization. Appending these six features to the original 19 gives
25 model inputs, retaining all original cube and task information. No raw or
processed stores need rewriting. Checkpoints save the state-key list, so
inference automatically uses the same feature representation.

Checks passed for all three colors, batched observations, expected relative
coordinates and exact offline/live feature parity on synthetic inputs. The
existing 19-input models retain their original feature keys. The new experiment
should use width 256, depth 2, seed 42 and a separate checkpoint directory.
The learner subsequently trained this representation; rollout results follow below.

### Goal-relative rollouts: epochs 30 and 60

The learner reported 0% for the best-validation checkpoint; no completed
sample count was supplied for that result. Visible failures included missed
red grasps, green hovering with an open gripper, and blue closing too high while
already leaving the cube. Later checkpoints showed more coordinated behavior.

| Epoch | Evaluation seed | Overall | Red | Green | Blue | Source |
|---:|---:|---:|---:|---:|---:|---|
| 30 | 789 | 4/30 (13%) | 1/10 | 2/10 | 1/10 | Learner terminal |
| 60 | 789 | 8/30 (27%) | 0/10 | 4/10 | 4/10 | Learner terminal |
| 60 | 987 | 13/90 (14%) | 1/30 | 8/30 | 4/30 | Learner terminal |
| 30 | 987 | 12/90 (13%) | 4/30 | 5/30 | 3/30 | Saved evaluation log |

The epoch-30 seed-987 evaluation completed with CUDA inference, shuffled layouts,
execution horizon 16 and an 800-step limit. The log is retained locally at
`logs/multicube/goal_relative/epoch030_seed987_90ep.log`.
Epoch 60 has only one more total success than epoch 30 on seed 987; this does
not establish a reliable overall advantage. Epoch 30 has more red successes in
these trials, while epoch 60 has more green and blue successes. Both checkpoints
remain useful references. Once seed 987 was used to compare checkpoints it
became another selection set; reserve a different seed for a final estimate
following additional tuning. Current evidence does not establish a general
improvement from relative features versus the original input representation.

### Grasp diagnostics and gripper-weighted loss

Across 42 candidate closure chunks, epoch 30 predicted a more-open mean gripper
command than the expert by more than 0.02 in 39 episodes, with overall mean
bias +0.118. Epoch 60 did so in 38 episodes, with mean bias +0.126. This is
command bias on expert observations, not measured jaw opening or rollout
success. It motivates a weighting experiment but does not establish the cause.

The learner retrained the 25-input, width-256, depth-2 model with normalized
squared-error weights [1, 1, 1, 4] divided by 7. Expert labels are unchanged;
this emphasizes gripper accuracy, including both closing and opening. Weighted
validation loss cannot be compared numerically with earlier unweighted MSE.
Outputs were saved separately in `checkpoints/multi_cube/goal_relative_gripper4/`.

Evaluated only the best-validation checkpoint, saved at epoch 2 with weighted
validation loss 0.544234, on seed 789: 30 shuffled multicube episodes, 10 per
color, maximum 800 steps and execution horizon 16, using CUDA inference.
Overall success was 2/30 (7%): red 0/10, green 2/10, blue 0/10. Full log is
`logs/multicube/goal_relative_gripper4/best_seed789_30ep.log`.
This small screening is not evidence of a reliable improvement. Later weighted
checkpoints have not been evaluated; comparing against the unweighted epoch-30
and epoch-60 results mixes checkpoint selection criteria and training duration.

### Gripper-weighted model: full checkpoint evaluation

Completed all ten periodic checkpoints on 30 episodes (10 per color) with
seed 789, reusing the completed best-validation evaluation. All runs used CUDA,
shuffled layouts, 800 maximum steps and execution horizon 16.

| Checkpoint | Overall | Red | Green | Blue |
|---|---:|---:|---:|---:|
| Best validation (epoch 2) | 2/30 | 0/10 | 2/10 | 0/10 |
| Epoch 10 | 5/30 | 0/10 | 4/10 | 1/10 |
| Epoch 20 | 5/30 | 1/10 | 3/10 | 1/10 |
| Epoch 30 | 6/30 | 1/10 | 2/10 | 3/10 |
| Epoch 40 | 1/30 | 0/10 | 0/10 | 1/10 |
| Epoch 50 | 1/30 | 0/10 | 0/10 | 1/10 |
| Epoch 60 | 2/30 | 0/10 | 1/10 | 1/10 |
| Epoch 70 | 5/30 | 0/10 | 3/10 | 2/10 |
| Epoch 80 | 3/30 | 0/10 | 2/10 | 1/10 |
| Epoch 90 | 4/30 | 0/10 | 2/10 | 2/10 |
| Epoch 100 | 4/30 | 0/10 | 2/10 | 2/10 |

Selected epoch 30, with 6/30 (20%) screening success, for a separate-seed
evaluation on seed 2468. Selection maximized overall success; ties would prefer
the strongest minimum per-color success and then the earlier epoch. The fresh
evaluation completed 90 episodes: **16/90 (17.8%)**, with red **3/30 (10%)**,
green **4/30 (13.3%)**, and blue **9/30 (30%)**.

The weighted epoch-60 model scored 2/30 on seed 789, compared with the earlier
unweighted epoch-60 model at 8/30 on the same scenarios. The weighted epoch-30
model scored 6/30 versus the earlier unweighted epoch-30 model at 4/30. These
small samples do not establish a reliable improvement from weighting. The
fresh weighted 16/90 on seed 2468 cannot be directly compared as a controlled
improvement against unweighted 13/90 on seed 987, because layouts differ.

Full logs, screening_results.json and confirmation_results.json are retained
locally under `logs/multicube/goal_relative_gripper4/full_comparison/`. Keep both
weighted and unweighted reference checkpoints; prioritize targeted grasp and
layout diagnostics before further changes to loss weights or model capacity.

### Layout coverage and geometric failure audit

Audited the 42 processed demonstration episodes and the actual 39-episode
training subset, excluding one-based validation episodes 37, 10 and 33. Slots
are numbered nearest-to-farthest along Y: 1=0.35 m, 2=0.45 m, 3=0.55 m,
4=0.70 m, each centered at X=-0.20 m. Pickup/bin pair is ordered.

| Target | Training pairs covered (out of 12) | Missing pickup → bin pairs |
|---|---:|---|
| red | 7 | 1 → 2, 1 → 4, 2 → 4, 3 → 1, 4 → 2 |
| green | 8 | 1 → 3, 1 → 4, 3 → 1, 3 → 2 |
| blue | 7 | 1 → 4, 2 → 3, 2 → 4, 3 → 4, 4 → 3 |

Repeated matched 30-episode evaluations on seed 789, collecting geometric
measurements from unweighted epoch 60 and weighted epoch 30. Success scores
reproduced 8/30 and 6/30 respectively. A substantial lift was defined as the
target cube rising at least 4 cm above its initial height. Reaching bin XY
was defined as target cube XY within 4 cm in each axis of the bin center at
any sampled step. These heuristic labels do not prove a secure grasp, correct
release timing or causality.

| Model | Success | No substantial lift | Lifted, never reached bin XY | Reached bin XY, failed placement |
|---|---:|---:|---:|---:|
| unweighted60 | 8 | 12 | 3 | 7 |
| weighted30 | 6 | 15 | 2 | 7 |

For layouts covered by training, unweighted epoch 60 achieved 6/22; for missing
pairs it achieved 2/8. Weighted epoch 30 achieved 6/22 and 0/8 respectively.
These small groups do not establish coverage as the sole explanation. Frequent
pickup failures also occur on demonstrated layout pairs. No simulation reset
or nonfinite-position event was detected by the diagnostic.

Coverage and per-episode telemetry are saved locally in
`logs/multicube/layout_diagnostics/coverage.json` and `rollouts.json`. The next
diagnostic should test imitation on one successful demonstration from its
recorded starting state, before assuming more capacity or more data is the fix.

### Single-demonstration replay and fitting diagnostic

Restored the recorded starting robot joint positions, all cube poses, bin center
and EE mocap pose for red demonstration 1. Velocities and warmstart were zeroed;
original velocity and mocap-target histories are not stored, so reconstruction
is approximate. Replaying processed actions through the existing apply_action
succeeded at step 324. EE path discrepancy averaged 4.0 mm, maximum 10.0 mm.
Recorded EE orientation changed by up to 6.94 degrees; XYZ replay kept the
initial mocap orientation. This success validates feasibility for this one
example, not all recordings or every initial-state detail.

A temporary 25-input width-256 depth-2 MLP was fitted on all complete 16-action
chunks of that single episode using unweighted normalized MSE, Adam 0.001 and
2,000 full-batch updates. Reference epoch-60 training normalization was reused
to avoid tiny single-episode variances. Best normalized training MSE remained
0.380975. Rollout from the reconstructed start failed after 800 steps; the cube
was above the bin at Z about 0.101 m. This finite fitting experiment does not
prove the model cannot fit the episode with other settings.

Per-action-dimension normalized MSE was approximately [0.304, 0.659, 0.539,
0.022] for X, Y, Z and gripper. Movement errors dominate this particular fit.
699 pairs of observations had normalized RMS distance <0.002 and different
future chunks with RMS difference >0.3. Example steps 76 and 77 differ by only
1.18e-5 in normalized observation RMS, but their future chunks differ by 1.945:
the same recorded movement bursts occur at shifted chunk offsets. These are
near-identical states, not exact duplicates or proof of irreducible error.
Human pause/key timing may contribute to hard-to-predict future chunks.

Diagnostic results, temporary weights and a near-observation example are saved
under `logs/multicube/single_demo_diagnostic/`. A shorter prediction chunk,
trained from scratch, is a motivated next experiment; merely truncating an
existing 16-action model is a different intervention.

### Single-demo observation diagnostics, horizon 16 retained

Compared the single-demo baseline with five added robot joint angles (30 inputs)
and a four-state history of the 25 normalized observations (100 inputs). History
was padded by repeating the first state and updated every control step during
rollout. Added joint features used statistics from the original 39 training
episodes. All fits used width 256, depth 2, seed 42, unweighted normalized MSE,
Adam 0.001, full batches, 2,000 updates and 16 predicted/executed actions.

| Observation | Best training MSE | Recorded-start rollout |
|---|---:|---|
| Current state (baseline) | 0.380975 | Failed |
| Current state plus 5 joint angles | 0.373721 | Cube out of bounds, step 227 |
| Last 4 states | 0.314428 | Failed, cube above bin at step 800 |

Baseline phase MSE by chunk start was approach 0.4324, closure 0.5598, carrying
0.3142, release 0.0290. Closure's vertical-motion error dominated that phase
(normalized Z MSE 2.0287). History reduced closure phase MSE to 0.3534 and Z MSE
to 1.2775 but did not produce a successful rollout. These phase labels come from
recorded geometry and command changes; chunks can straddle phase boundaries.

The initial joint-model diagnostic ran beyond cube loss and later produced a
MuJoCo numerical instability. Its post-reset path measurements are invalid.
The saved weights were rechecked with normal out-of-bounds termination; the
cube exited bounds at step 227. No retraining was performed for that correction.

Results and isolated diagnostic weights are retained under
`logs/multicube/single_demo_state_tests/`. These are one-episode diagnostics,
not aggregate success rates. History improves fit for this example but does
not establish a general solution. A shorter prediction-horizon diagnostic
remains a motivated next step before training the full dataset again.

### Short-horizon and absolute-position diagnostics

The four-action single-red-demo model was trained from scratch with the same
25 inputs, width 256, depth 2, seed 42, reference normalization, unweighted MSE,
Adam 0.001 and 2,000 full-batch updates. Training MSE was 0.370960 versus
0.380975 for the 16-action baseline, but the rollout did not lift the cube
(max Z 0.0211 m). This test does not support shorter prediction as the fix.
Different horizons also have different targets and counts of valid chunks.
Results are under `logs/multicube/single_demo_horizon4/`.

Next tested predicting absolute future EE XYZ rather than per-step deltas, with
absolute gripper commands unchanged. Future positions were reconstructed from
current position plus cumulative expert deltas. XYZ output normalization used
reference EE-position statistics; gripper used reference action statistics.
This changes both target representation and movement loss scaling. Its MSE
therefore cannot be directly compared with delta-action normalized MSE.
Execution set absolute mocap positions; prediction/execution horizons stayed 16.
All tests used the same width/depth, seed, Adam and 2,000-update fitting budget.

| Training collection | Recorded-start outcomes |
|---|---|
| Red demonstration 1 only | Success at step 274 |
| Green demonstration 6 only | Cube left bounds at step 186 |
| Blue demonstration 21 only | Failed at step 800, cube above bin |
| Demonstrations 1, 6, 21 together | Red success at step 301; green failed; blue left bounds |

The single-red absolute model's future-position RMSE was 2.59 mm versus 2.36 mm
for integrated predictions of the delta model on the same expert observations.
Thus success cannot be explained by lower offline position RMSE alone; direct
position execution changes feedback and drift. The successful red case is not
a generalization success estimate or proof of superiority on the full task.

Replayed expert delta actions separately for all three recorded initial
configurations: red, green and blue all succeeded (steps 324, 307 and 205).
Approximate initialization limitations remain as previously described. Failures
of the single-green and single-blue learned models therefore are not specific
to combining target colors, although exact causal mechanisms remain unresolved.

Results and isolated diagnostic weights are under
`logs/multicube/single_demo_absolute_positions/` and
`logs/multicube/three_demo_absolute_positions/`. Raw demonstrations, original
processed stores and production checkpoints were not altered. Absolute-output
diagnostic weights require their special output normalization and direct-target
execution; they are not interchangeable with delta-policy eval checkpoints.

### Recorded versus live observations; multicube DAgger preparation

Reused the isolated absolute-position single-demo models, with identical weights,
starting configuration and 16-action execution. Recorded-input runs supplied the
expert's observation at each prediction time, while physics evolved under the
model's actions. These are privileged diagnostic inputs, not policy success-rate
evaluations. Recorded-input runs stop at demonstration length; live runs allow
800 steps. All expert-fed successes occurred within the recorded interval.

| Single-demo model | Recorded observations | Live observations |
|---|---|---|
| Red episode 1 | Success, step 338 | Success, step 274 |
| Green episode 6 | Success, step 322 | Cube out of bounds, step 186 |
| Blue episode 21 | Success, step 210 | Failed at step 800 |

Recorded-input mean EE path errors were approximately 5.8, 8.7 and 7.1 mm for
red, green and blue. The comparison supports feedback/distribution drift as a
contributor in these examples; it does not prove the sole cause or automatically
transfer to every full-data policy. DAgger corrections from learner-visited states
are now a motivated next experiment. Results are retained locally under
`logs/multicube/recorded_vs_live/results.json`.

Extended `scripts/dagger_eval.py` with --multicube and --goal-cube. Multicube
recordings retain all cube poses, one-hot goal and bin position; default outputs
use datasets/raw/multi_cube/dagger/<timestamp>. Targets cycle red/green/blue
when --goal-cube all is used and appear on the display. Human handback finalizes
a contiguous segment so policy gaps cannot become synthetic expert deltas.
Unsuccessful current segments are discarded on cube-out-of-bounds termination.
Existing single-cube behavior is retained.

A temporary simulated-key test checked takeover, handback, a subsequent takeover
and Escape discard: one saved 21-row segment retained constant green labels,
all cube/bin arrays and aligned episode boundaries. CLI help and whitespace
checks passed. No expert corrections were added by these tests. Collection
should use the existing full-data unweighted epoch-60 delta policy, rather than
the special absolute-output single-demo diagnostics, for a controlled comparison.

### ACT scene tokens and action-representation comparison

The full multicube collection used here contains 42 original demonstrations
(11 red, 15 green, 16 blue) and seven DAgger correction segments: 11,601 processed
transitions. Validation retains original zero-based episodes [36, 9, 32]; the
46 training segments provide 10,127 chunks and validation provides 739 chunks.
The earlier MLP plus seven corrections reached 10/30 on several screening
checkpoints; this is not a matched fresh-seed comparison with ACT.

The scene-token model uses 25 observations: EE XYZ, gripper, three cube XYZ
positions, goal one-hot, bin XYZ, target-relative XYZ and bin-relative XYZ.
Eight scene tokens plus a latent token feed its transformer decoder. Training
uses width 256, depth 2, four attention heads, latent dimension 32, MSE
reconstruction, KL weight 0.1, 100 epochs, batch size 64 and seed 42. Adam starts
at 1e-4 with cosine decay to 1e-5. Prediction and execution horizons are 16.
This compact state-based ACT adaptation differs from the original ACT setup.

| Epoch | Delta scene ACT /30 | Absolute-position scene ACT /30 |
|---:|---:|---:|
| 10 | 3 | 0 |
| 20 | 8 | 0 |
| 30 | 11 | 0 |
| 40 | 12 | 0 |
| 50 | 14 | 1 |
| 60 | 15 | 0 |
| 70 | 14 | 2 |
| 80 | 16 | 4 |
| 90 | 13 | 2 |
| 100 | 14 | 1 |

Screening uses seed 789, 30 episodes (10 per color), an 800-step limit and CUDA.
Both variants select epoch 80. Separate-seed confirmation uses seed 2026 and
90 episodes (30 per color): delta 34/90 (red 11, green 13, blue 10), absolute
0/90. Confirmation for the prior flat-state ACT at KL 0.1 was learner-reported
as 22/90; the scene model is the retained reference. These are single-training-
seed experiments, not evidence of architecture superiority in general.

Reference checkpoint:
`checkpoints/multi_cube/act_scene_dagger7_mse_kl01/epoch080_ee_xyz_act_scene_dagger7ep_width256.pt`.
Absolute outputs reconstruct next measured EE XYZ as state_ee_xyz + action_ee_xyz
and set the mocap target directly. Gripper targets are unchanged. Expert replay
succeeded for one demonstration per color, with approximate initial-state
reconstruction. The learned model's failure therefore warrants distinction from
basic replay feasibility. Normalized losses across representations are not
directly comparable.

Retained local results:
- `logs/multicube/act_scene_dagger7_mse_kl01/evaluation_HIsgfP/`
- `logs/multicube/act_scene_dagger7_absolute_kl01/`

### 2026-10-16: prepare 24 targeted ACT corrections

Repeated the selected delta checkpoint's 30-episode seed-789 evaluation using
the existing evaluation loop, adding observation logging only. It reproduced
16/30, with red 7/10, green 3/10 and blue 6/10.

| Outcome | Episodes |
|---|---:|
| Success | 16 |
| No substantial target lift | 2 |
| Lifted, never reached bin XY | 2 |
| Reached bin XY, failed placement | 10 |

Lift means at least 4 cm above initial target height; reaching bin XY means
within 4 cm on each horizontal axis at any sampled step. These categories do
not prove secure grasp, release failure or causality. Several failed episodes
ended with the target elevated near the gripper above the bin. No non-target
cube was detected inside the bin during these 30 rollouts. This does not rule
out other forms of incorrect target interaction.

Refreshed training coverage includes 12 red, 17 green and 17 blue segments.
Nearest-slot pickup/bin pairs at segment start cover 8/12 red, 9/12 green and
8/12 blue combinations. DAgger segments begin at takeover; their inferred slot
is not necessarily the original pickup slot.

| Color | Missing segment-start pickup → bin pairs |
|---|---|
| red | 1 → 2, 1 → 4, 2 → 4, 4 → 2 |
| green | 1 → 4, 3 → 1, 3 → 2 |
| blue | 1 → 4, 2 → 4, 3 → 4, 4 → 3 |

Fixed DAgger's success grace period: losing success resets the countdown rather
than allowing a stale timer to save a failed placement as successful. After
first detection, require 17 further successful control steps at 10 Hz. Checks
passed for sustained success, lost/recovered success, prolonged failure and
out-of-bounds discard. This changes recording, not the evaluation success rule;
human inspection must still verify that the cube is released and settled.

Next collection: eight successful contiguous recoveries per color using the
reference checkpoint and seed 31415, with bin approach/release as the main
priority and pickup/transport corrections when needed. Hold human control
through release and automatic settling/save. Record failures and layouts, and
curate before preprocessing. The 24 corrections have not yet been collected.

Then train from scratch with the same settings and validation IDs, append the
new data after the existing two stores and save to a distinct directory. Select
among every-tenth-epoch checkpoints using seed 789 (30 episodes). Freeze that
choice before evaluating it and the retained reference on seeds 4242 and 4243,
90 episodes each. Report total/per-color counts and uncertainty. Collection
results determine the next experiment; do not assume reaching 50% in advance.

Diagnostic coverage, per-step rollout traces and the audit script are retained
in `logs/multicube/act_scene_dagger24_preparation/` (ignored local artifacts).

### 2026-10-23: curate and preprocess the next DAgger collection

Inspected four new raw sessions: the first was empty after restart; the others
contained 8 red, 10 green and 13 blue segments. In the blue session
`2026-10-23_16-55-01`, one-based segment 6 contained only four steps and segment 7
ended without placement. These two were excluded from a separate curated copy;
all raw recordings remain unchanged. This exclusion follows this round's
complete-successful-recovery selection rule, not a claim that every incomplete
DAgger segment is inherently unusable.

Retained 29 corrections: 8 red, 10 green, 11 blue. Each ends with the target
geometrically inside the bin for 17 recorded steps; goal labels are constant
and all recorded arrays are finite. Geometry alone does not prove release from
the claw. The curated stores contain 1,502 observations; per-episode delta
computation produces 1,473 transitions. Verified every XYZ delta against
consecutive raw EE positions, gripper command alignment, array lengths and
episode boundaries.

Curated collection: `datasets/curated/multi_cube/dagger_2026-10-23/`.
Processed output: `datasets/processed/multi_cube/dagger_new29_ee_xyz.zarr`.
Curation manifest and audit scripts:
`logs/multicube/dagger_new29_preparation/`.

The next training collection has 42 original episodes + 7 earlier correction
segments + 29 new segments: 78 episodes/segments and 13,074 transitions. With
validation IDs [36, 9, 32] retained, 75 segments train and three validate.
At horizon 16, the new data adds 1,038 complete chunks. Training and evaluation
of this expanded collection are pending; no success-rate improvement is yet
claimed.

### 2026-10-23: ACT with 36 DAgger corrections — completed evaluation

Retraining on 42 original demonstrations plus 36 corrections completed. The
epoch comparison selected epoch 90 at 22/30 on seed 789. This selection was
frozen before confirmation: seed 4242 achieved 54/90 (red 17, green 15, blue 22),
and seed 4243 achieved 58/90 (red 23, green 16, blue 19). Combined: **112/180
(62.2%)**, red 40/60, green 31/60, blue 41/60. Selected validation MSE: 0.917494.

The complete architecture, epoch table, reproduction commands and interpretation
are consolidated in [the ACT report](act.md). Published machine-readable results
are in [CSV](results/act-dagger36.csv) and [JSON](results/act-dagger36.json).
Earlier entries describing this run as pending record its preparation stage;
this entry records the completed run. No matched old-policy confirmation on
seeds 4242/4243 has been performed, so the earlier 34/90 is a different-seed
reference rather than a matched estimate of improvement.
