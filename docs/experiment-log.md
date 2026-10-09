# Experiment log

## Initial data collection

- Task: pick up the cube, move around the obstacle, release it into the bin.
- Collection: 22 saved demonstrations in the final recording session.
- Selection: exclude original episode 13 (no release), retain 21 episodes.
- Cleaned observations: 11,894.
- Baseline observation plan: EE XYZ, gripper opening, cube XYZ, obstacle XYZ.
- Baseline action plan: EE XYZ deltas and absolute gripper command.
- Model plan: MLP predicting action chunks with mean squared error.

## Baseline status

Not trained. Architecture, optimization settings, episode split, checkpoint,
validation loss, and rollout success rate have not been finalized or measured.
The supplied train/model files still contain TODOs.

For each trained experiment, record: data selection, train/validation episode IDs,
seed, state/action keys, chunk length, architecture, optimizer settings,
checkpoint identity, validation loss, and evaluation successes/episode count.

The keyboard demonstration in `media/` is a successful scripted replay of the
teleoperation commands. It is not evidence of learned-policy performance.
