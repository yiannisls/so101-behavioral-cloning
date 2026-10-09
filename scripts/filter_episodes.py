"""Copy selected complete episodes to a new Zarr store; preserve raw recordings.

Episode numbers are one-based within the input store, matching the recording
counter after an episode is saved. All data arrays and boundaries stay aligned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def filter_episodes(input_path: Path, output_path: Path, excluded: list[int]) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if output_path == input_path or output_path.is_relative_to(input_path):
        raise ValueError("Output must be a separate store outside the input store.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    source = zarr.open_group(str(input_path), mode="r")
    ends = np.asarray(source["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    if not len(ends) or np.any(ends <= starts):
        raise ValueError("Input must contain complete, nonempty episodes.")
    excluded_set = set(excluded)
    invalid = excluded_set - set(range(1, len(ends) + 1))
    if invalid:
        raise ValueError(f"Episode numbers outside 1..{len(ends)}: {sorted(invalid)}")
    kept = [n for n in range(1, len(ends) + 1) if n not in excluded_set]
    if not kept:
        raise ValueError("At least one episode must be retained.")
    if set(source.group_keys()) != {"data", "meta"}:
        raise ValueError("Expected homework data/meta groups.")
    if set(source["meta"].array_keys()) != {"episode_ends"}:
        raise ValueError("Unexpected metadata arrays; inspect their alignment before filtering.")

    indices = np.concatenate([np.arange(starts[n - 1], ends[n - 1]) for n in kept])
    new_ends = np.cumsum([ends[n - 1] - starts[n - 1] for n in kept], dtype=np.int64)
    names = list(source["data"].array_keys())
    for name in names:
        if source["data"][name].shape[0] < ends[-1]:
            raise ValueError(f"Array {name} does not cover all completed episodes.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    destination = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    destination.attrs.update(dict(source.attrs))
    destination.attrs.update({
        "source_zarr": str(input_path),
        "source_episode_numbers": kept,
        "excluded_source_episode_numbers": sorted(excluded_set),
        "num_episodes": len(kept),
        "num_recorded_steps": int(new_ends[-1]),
    })
    data = destination.create_group("data")
    data.attrs.update(dict(source["data"].attrs))
    for name in names:
        original = source["data"][name]
        selected = np.asarray(original[:])[indices]
        copied = data.create_array(
            name, data=selected, chunks=original.chunks,
            compressors=original.compressors,
        )
        copied.attrs.update(dict(original.attrs))
        # Verify data integrity, including the splice where the removed episode was.
        np.testing.assert_array_equal(copied[:], selected)
        print(f"  {name}: {original.shape} -> {copied.shape}")

    meta = destination.create_group("meta")
    meta.attrs.update(dict(source["meta"].attrs))
    original_ends = source["meta"]["episode_ends"]
    copied_ends = meta.create_array(
        "episode_ends", data=new_ends, chunks=original_ends.chunks,
        compressors=original_ends.compressors,
    )
    copied_ends.attrs.update(dict(original_ends.attrs))
    np.testing.assert_array_equal(copied_ends[:], new_ends)
    print(f"Kept {len(kept)}/{len(ends)} episodes; excluded {sorted(excluded_set)}")
    print(f"Removed {int(ends[-1] - new_ends[-1])} recorded steps")
    print(f"Cleaned copy: {output_path}")
    print("Original recording unchanged.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-episodes", type=int, nargs="+", required=True,
                        help="One-based episode numbers within the input store.")
    args = parser.parse_args()
    filter_episodes(args.input, args.output, args.exclude_episodes)


if __name__ == "__main__":
    main()
