# The cube currently is being left at the left wall of the box, and i believe
# its because when i was training the dataset, i never center the gripper
#to open inside of the center of the box, but always on the left,
# leaving a small margin

#This is a first diagnostic, measuring the start of opening rather
# than the exact instant the cube leaves the fingers.
# If you moved while gradually opening, we may need to
#
#  inspect that interval too.
#For scale, the bin’s inner half-width is about 5.5 cm,
# and the cube’s half-width is 2 cm. An upright cube therefor
# e needs its center within roughly 3.5 cm of the bin center on
#  each axis to clear the walls. Rotation can reduce that margin.

#FROM
#     17 |          0.11 |         -2.76
#     18 |          1.31 |         -1.34
#     19 |         -1.31 |         -1.80
#     20 |          0.52 |         -2.75
#     21 |          0.21 |         -2.15
#then recorded 5 new episodes with the gripper centered in the bin, and got:
#1-behavior-cloning$ python scripts/check_releases.py
#Episode | X offset (cm) | Y offset (cm)
#      1 |          1.14 |         -1.26
#      2 |          0.95 |         -1.61
#      3 |         -0.34 |         -1.79
#      4 |         -0.45 |         -1.48
#      5 |          0.08 |         -1.12
import numpy as np
import zarr

root = zarr.open_group(
    "/home/ylspyrou/Desktop/so101-behavior-cloning/datasets/raw/single_cube/teleop/2026-10-11_15-20-46/so100_transfer_cube_teleop.zarr",
    mode="r",
)

cube = np.asarray(root["data"]["state_cube"][:])
command = np.asarray(root["data"]["action_gripper"][:]).reshape(-1)
ends = np.asarray(root["meta"]["episode_ends"][:])
starts = np.concatenate(([0], ends[:-1]))

bin_xy = np.array([-0.20, 0.70])

print("Episode | X offset (cm) | Y offset (cm)")

for episode, (start, end) in enumerate(zip(starts, ends), start=1):
    positions = cube[start:end, :3]
    gripper_commands = command[start:end]

    # A positive change means you commanded the gripper to open.
    opening = np.concatenate(([False], np.diff(gripper_commands) > 0.01))

    # Ignore opening at the beginning, far from the bin.
    near_bin = np.all(
        np.abs(positions[:, :2] - bin_xy) < 0.10,
        axis=1,
    )
    raised = positions[:, 2] > 0.04

    candidates = np.flatnonzero(opening & near_bin & raised)

    if len(candidates) == 0:
        print(f"{episode:7d} | opening not detected")
        continue

    # Position at the first opening command near the bin.
    t = candidates[0]
    offset_cm = 100 * (positions[t, :2] - bin_xy)

    print(
        f"{episode:7d} | {offset_cm[0]:13.2f} | {offset_cm[1]:13.2f}"
    )
