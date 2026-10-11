import torch
from hw3.model import build_policy

policy = build_policy(
    policy_type="obstacle",
    state_dim=10,
    action_dim=4,
    chunk_size=16,
)

# Fake batch: 32 observations, each containing 10 values.
states = torch.randn(32, 10)

# Fake expert targets: 16 actions of 4 values per observation.
expert_chunks = torch.randn(32, 16, 4)

predictions = policy.sample_actions(states)
loss = policy.compute_loss(states, expert_chunks)
loss.backward()

print("Prediction shape:", predictions.shape)
print("Loss:", loss.item())
print(
    "Gradient exists:",
    policy.network[0].weight.grad is not None,
)
