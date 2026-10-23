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
from hw3.model import ACTPolicy, BasePolicy, build_policy

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
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0}
    n_samples = 0

    for states, action_chunks in loader:
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        optimizer.zero_grad()
        if isinstance(model, ACTPolicy):
            losses = model.compute_loss_components(states, action_chunks)
        else:
            losses = {"total": model.compute_loss(states, action_chunks)}

        losses["total"].backward()
        optimizer.step()

        batch_size = states.shape[0]
        n_samples += batch_size

        for name, value in losses.items():
            totals[name] += value.detach().item() * batch_size

    return {
        name: value / max(n_samples, 1)
        for name, value in totals.items()
    }

@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0

    for states, action_chunks in loader:
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        if isinstance(model, ACTPolicy):
            predictions = model.sample_actions(states)  # z = 0
            loss = torch.nn.functional.mse_loss(
                predictions, action_chunks
            )
        else:
            loss = model.compute_loss(states, action_chunks)

        batch_size = states.shape[0]
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / max(n_samples, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask", "act", "act_scene"],
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
    parser.add_argument("--d-model", type=int, default=256,
                        help="Hidden layer width (default: 256).")
    parser.add_argument("--depth", type=int, default=2,
                        help="Number of hidden layers (default: 2).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Optional directory for this experiment's checkpoints.")
    parser.add_argument(
        "--val-episode-ids", type=int, nargs="+", default=None,
        help="Explicit zero-based validation episode IDs in merged dataset order.",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=10.0,
        help="ACT latent regularization weight (default: 10.0).",
    )
    args = parser.parse_args()
    if args.d_model < 1 or args.depth < 1:
        parser.error("--d-model and --depth must both be positive.")

    if args.policy == "act_scene":
        expected_keys = [
            "state_ee_xyz",
            "state_gripper",
            "original_pos_cube_red[:3]",
            "original_pos_cube_green[:3]",
            "original_pos_cube_blue[:3]",
            "state_goal",
            "goal_pos",
            "goal_relative",
        ]

        if args.state_keys is None:
            args.state_keys = expected_keys
        elif args.state_keys != expected_keys:
            parser.error(
                "act_scene requires this state-key order: "
                + " ".join(expected_keys)
            )

    if args.policy in ("multitask", "act", "act_scene"):
        if args.state_keys is None:
            args.state_keys = [
                "state_ee_xyz", "state_gripper",
                "original_pos_cube_red[:3]",
                "original_pos_cube_green[:3]",
                "original_pos_cube_blue[:3]",
                "state_goal", "goal_pos",
            ]
        if args.action_keys is None:
            args.action_keys = ["action_ee_xyz", "action_gripper"]

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
    if args.val_episode_ids is not None:
        if (len(set(args.val_episode_ids)) != len(args.val_episode_ids)
                or any(i < 0 or i >= num_episodes for i in args.val_episode_ids)
                or len(args.val_episode_ids) >= num_episodes):
            raise ValueError("Validation IDs must be unique, in range, and leave training episodes.")
        val_episode_ids = torch.tensor(args.val_episode_ids, dtype=torch.long)
    elif args.policy in ("multitask", "act", "act_scene"):
        # Keep complete episodes of every target color out of training.
        episode_goals = []
        for zp in zarr_paths:
            root = zarr_lib.open_group(str(zp), mode="r")
            ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
            goals = np.asarray(root["data"]["state_goal"][:])
            starts = np.concatenate(([0], ends[:-1]))
            for start, end in zip(starts, ends):
                goal = goals[int(start):int(end)]
                if not (
                    np.all(goal == goal[0])
                    and np.all((goal[0] == 0) | (goal[0] == 1))
                    and goal[0].sum() == 1
                ):
                    raise ValueError("Each episode must have one constant one-hot goal.")
                episode_goals.append(int(goal[0].argmax()))
        episode_goals = torch.tensor(episode_goals)
        validation_ids = []
        for color_id, color in enumerate(("red", "green", "blue")):
            color_ids = episode_order[episode_goals[episode_order] == color_id]
            if len(color_ids) < 2:
                raise ValueError(f"Need at least two {color} episodes for train/validation.")
            n_holdout = max(1, int(len(color_ids) * VAL_SPLIT))
            validation_ids.append(color_ids[:n_holdout])
            print(f"{color}: {len(color_ids) - n_holdout} train / {n_holdout} validation episodes")
        val_episode_ids = torch.cat(validation_ids)
    else:
        # Retain the original single-cube experiment's fixed validation split.
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
        d_model=args.d_model,
        depth=args.depth,
        kl_weight=args.kl_weight,
    ).to(device)

    if isinstance(model, ACTPolicy):
        print(f"ACT KL weight: {model.kl_weight}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # DONE: implement an optimizer and scheduler
    learning_rate = (
        1e-4 if isinstance(model, ACTPolicy) else LR
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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

    ckpt_dir = args.checkpoint_dir or (ckpt_dir / "epoch_comparison")

    save_name = save_name.replace(
        ".pt",
        f"_width{model.d_model}.pt",
    )

    if model.depth != 2:
        save_name = save_name.replace(".pt", f"_depth{model.depth}.pt")

    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        train_loss = train_metrics["total"]
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
                    "latent_dim": getattr(model, "latent_dim", 32),
                    "nhead": getattr(model, "nhead", 4),
                    "kl_weight": getattr(model, "kl_weight", 10.0),
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "val_loss": val_loss,
                },
                save_path,
            )
            tag = " ✓ saved"

        details = ""
        if isinstance(model, ACTPolicy):
            details = (
                f" | recon {train_metrics['reconstruction']:.6f}"
                f" | KL {train_metrics['kl']:.6f}"
            )
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {train_loss:.6f}{details} | "
            f"val {val_loss:.6f}{tag}"
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
