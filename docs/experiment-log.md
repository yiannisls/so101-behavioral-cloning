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
