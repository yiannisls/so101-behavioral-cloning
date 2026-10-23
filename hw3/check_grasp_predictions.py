"""Compare expert and policy chunks near candidate grasp-closure events."""
import argparse
from pathlib import Path

import numpy as np
import torch
import zarr

from hw3.dataset import load_zarr
from hw3.eval_utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, nargs='+', default=[1, 6, 21],
                        help='One-based demonstration numbers (default: first of each color).')
    args = parser.parse_args()
    path = Path('datasets/processed/multi_cube/processed_ee_xyz.zarr')
    root = zarr.open_group(str(path), mode='r')
    data = root['data']
    ends = np.asarray(root['meta']['episode_ends'][:], dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    goals = np.asarray(data['state_goal'][:])
    ee = np.asarray(data['state_ee_xyz'][:])
    cubes = np.stack([
        np.asarray(data[f'original_pos_cube_{c}'][:])[:, :3]
        for c in ('red', 'green', 'blue')
    ], axis=1)
    grip = np.asarray(data['action_gripper'][:])[:, 0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in (30, 60):
        ckpt = Path(f'checkpoints/multi_cube/goal_relative/epoch{epoch:03d}_ee_xyz_multitask_width256.pt')
        model, norm, horizon, state_keys, action_keys = load_checkpoint(ckpt, device)
        if action_keys != ['action_ee_xyz', 'action_gripper']:
            raise ValueError('This diagnostic expects XYZ deltas followed by absolute gripper control.')
        states, actions, _ = load_zarr(path, state_keys, action_keys)
        for episode in args.episodes:
            if not 1 <= episode <= len(ends):
                raise ValueError(f'Episode must be between 1 and {len(ends)}.')
            start, end = int(starts[episode - 1]), int(ends[episode - 1])
            color_id = int(goals[start].argmax())
            distance = np.linalg.norm(ee[start:end] - cubes[start:end, color_id], axis=1)
            # Lower control commands close this simulator's gripper.
            change = np.diff(grip[start:end], prepend=grip[start])
            candidates = np.flatnonzero((change < -0.03) & (distance < 0.12))
            candidates = candidates[candidates + horizon <= end - start]
            if not len(candidates):
                print(f'Episode {episode}: no candidate closure event with a full chunk.')
                continue
            row = start + int(candidates[0])
            state = torch.as_tensor(norm.normalize_state(states[row]), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                predicted = model.sample_actions(state)[0].cpu().numpy()
            predicted = norm.denormalize_action(predicted)
            expert = actions[row:row + horizon]
            color = ('red', 'green', 'blue')[color_id]
            print(f'\nEpoch {epoch} | Episode {episode} ({color}) | Candidate closure step {row-start}')
            print('Offset | Expert dZ(cm) | Model dZ(cm) | Expert grip | Model grip')
            for offset in range(horizon):
                print(f'{offset:6d} | {100*expert[offset,2]:13.3f} | {100*predicted[offset,2]:12.3f} | {expert[offset,3]:11.3f} | {predicted[offset,3]:10.3f}')
            print(f'Chunk net dZ: expert {100*expert[:,2].sum():.2f} cm, model {100*predicted[:,2].sum():.2f} cm')
    print('\nThese predictions start from expert observations; this is not a rollout or a success-rate evaluation.')


if __name__ == '__main__':
    main()
