"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
import numpy as np
from torch.utils.data import DataLoader, Subset

# TODO: Choose your own hyperparameters!
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
VAL_SPLIT = 0.1


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        # TOD: Implement the training step for one batch here.

        optimizer.zero_grad()
        loss = model.compute_loss(states, action_chunks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        # This mostly: Get states and action_chunks onto the correct device, compute the loss, and step the optimizer.

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        # Implement the evaluation step for one batch here.
        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--extra-zarr",
        type=Path,
        nargs="+",
        default=None,
        help="Additional processed datasets to merge.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )

    # Separate the number of episodes for training and validation
    num_episodes = len(ep_ends)
    n_val_episodes = max(1, int(num_episodes * VAL_SPLIT))

    episode_order = torch.randperm(
        num_episodes,
        generator=torch.Generator().manual_seed(args.seed),
    )
    # val_episode_ids = episode_order[:n_val_episodes]
    # train_episode_ids = episode_order[n_val_episodes:]
    val_episode_ids = torch.tensor([15, 8])
    train_episode_ids = episode_order[
        ~torch.isin(episode_order, val_episode_ids)
    ]

    print("Training episode IDs:", train_episode_ids.tolist())
    print("Validation episode IDs:", val_episode_ids.tolist())

    # Instead of normalizer, we can use the training episode IDs
    # to create a mask for the training data.
    # This way, we can compute the normalizer using only the
    # training data and avoid any data leakage from the validation set.

    row_episode_ids = np.searchsorted(
        ep_ends,
        np.arange(len(states)),
        side="right",
    )

    training_row_mask = np.isin(
        row_episode_ids,
        train_episode_ids.numpy(),
    )

    normalizer = Normalizer.from_data(
        states[training_row_mask],
        actions[training_row_mask],
    )
    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # ── train / val split ─────────────────────────────────────────────
    #Replace random split wth a split based on episode IDs
    # to avoid data leakage between train and val sets.
    chunk_episode_ids = np.searchsorted(
        ep_ends,
        dataset.indices,
        side="right",
    )

    train_indices = np.flatnonzero(
        np.isin(chunk_episode_ids, train_episode_ids.numpy())
    )

    val_indices = np.flatnonzero(
        np.isin(chunk_episode_ids, val_episode_ids.numpy())
    )

    train_ds = Subset(dataset, train_indices.tolist())
    val_ds = Subset(dataset, val_indices.tolist())

    print(f"Training chunks: {len(train_ds)}")
    print(f"Validation chunks: {len(val_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        d_model=256,
        depth=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # DONE: implement an optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")

    ckpt_dir = ckpt_dir / "epoch_comparison"

    save_name = save_name.replace(
        ".pt",
        f"_width{model.d_model}.pt",
    )

    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": args.chunk_size,
                    "policy_type": args.policy,
                    "d_model": model.d_model,
                    "depth": model.depth,
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "val_loss": val_loss,
                },
                save_path,
            )
            tag = " ✓ saved"


        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )
        if epoch % 10 == 0:
            checkpoint = torch.load(
                save_path,
                map_location="cpu",
                weights_only=False,
            )
            checkpoint.update(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                val_loss=val_loss,
            )

            epoch_path = save_path.with_name(
                save_path.name.replace(
                    "best_model_",
                    f"epoch{epoch:03d}_",
                    1,
                )
            )
            torch.save(checkpoint, epoch_path)
            print(f"Periodic checkpoint: {epoch_path}")

    # The epoch loop ends here.
    last_checkpoint = torch.load(
        save_path,
        map_location="cpu",
        weights_only=False,
    )
    last_checkpoint.update(
        epoch=epoch,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        val_loss=val_loss,
    )

    last_path = save_path.with_name(
        save_path.name.replace("best_model_", "last_model_", 1)
    )
    torch.save(last_checkpoint, last_path)
    print(f"Final-epoch checkpoint: {last_path}")

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
