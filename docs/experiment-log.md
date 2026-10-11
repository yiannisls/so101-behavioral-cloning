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

## Exercise 2: DAgger starting point

Retain the 61% baseline. Evaluate it under adversarial obstacle placement,
observe failures, collect expert corrections with `dagger_eval.py`, preprocess,
and retrain. No DAgger results have been measured yet. The default-distribution
baseline and adversarial-distribution evaluation must be reported separately.
