"""Shared goal-relative features for offline data and live observations."""

import numpy as np


def goal_relative_features(robot_xyz, red_xyz, green_xyz, blue_xyz, goal, bin_xyz):
    """Return target-minus-robot and bin-minus-robot XYZ, before normalization."""
    cube_positions = np.stack([red_xyz, green_xyz, blue_xyz], axis=-2)
    target_xyz = np.sum(
        cube_positions * np.asarray(goal)[..., :, None], axis=-2
    )
    return np.concatenate(
        [target_xyz - robot_xyz, bin_xyz - robot_xyz], axis=-1
    ).astype(np.float32)


if __name__ == "__main__":
    print(goal_relative_features(
        np.array([0., 0., 0.]),
        np.array([.1, 0., 0.]),
        np.array([0., .2, 0.]),
        np.array([0., 0., .3]),
        np.array([0, 1, 0]),
        np.array([.4, .5, 0.]),
    ))
