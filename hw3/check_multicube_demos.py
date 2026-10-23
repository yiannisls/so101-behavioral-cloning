import numpy as np
import zarr

root = zarr.open_group(
    "datasets/processed/multi_cube/processed_ee_xyz.zarr",
    mode="r",
)
data = root["data"]
ends = np.asarray(root["meta"]["episode_ends"][:])
starts = np.r_[0, ends[:-1]]
colors = ["red", "green", "blue"]

cubes = np.stack([
    np.asarray(data[f"original_pos_cube_{color}"][:])[:, :3]
    for color in colors
], axis=1)
goals = np.asarray(data["state_goal"][:])
bins = np.asarray(data["goal_pos"][:])

print("Episode | Target | Cubes inside bin at final recorded step")
for episode, (start, end) in enumerate(zip(starts, ends), 1):
    start, end = int(start), int(end)
    target = colors[int(goals[start].argmax())]
    final = cubes[end - 1]
    inside = (
        np.all(np.abs(final[:, :2] - bins[end - 1, :2]) < 0.04, axis=1)
        & (final[:, 2] > 0)
        & (final[:, 2] < 0.04)
    )
    placed = [colors[i] for i in range(3) if inside[i]]
    print(f"{episode:7d} | {target:6s} | {placed}")
