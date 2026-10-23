# Sources and attribution

## ETH Zürich Robot Learning course

The original HW3 imitation-learning assignment supplies the MuJoCo simulation,
teleoperation tools, dataset utilities, action preprocessing, evaluation harness,
and policy/training templates. This project is an independent study built on that
material and is not an official ETH project.

Original source: https://github.com/tenvinc/ethz-course-2026
HW3: https://github.com/tenvinc/ethz-course-2026/tree/main/hw3_imitation_learning
Upstream revision inspected for provenance: `2e089e97d84968182c1ffa2e1bc8177530feb6fe`.

No top-level license is present in the upstream repository at that revision.
The course material retains its original authorship. Its public availability
does not establish a permissive software license, and no blanket license has
been applied to the supplied code here. The explicit Apache-2.0 license for the
robot assets is preserved separately below.
See `docs/course-assignment.md` for the supplied assignment.

## Robot model and meshes

The bundled Standard Open Arm-100 description is supplied under Apache-2.0.
Its original README and license are retained at
`so101_gym/assets/trs_so_arm100/README.md` and
`so101_gym/assets/trs_so_arm100/LICENSE`.
The asset README credits The Robot Studio's SO-ARM100 project:
https://github.com/TheRobotStudio/SO-ARM100

## Work added in this study

- Demonstration collection, curation, behavioral cloning and DAgger experiments.
- Implemented policy models, training/validation logic, architecture-aware
  checkpoints, goal-relative features and multicube takeover recording.
- A state-based ACT adaptation with separate decoder scene tokens and a
  variational action encoder, using PyTorch transformer components.
- Diagnostics, experiment reports, published result summaries and rollout media.

ACT algorithm reference: Tony Z. Zhao, Vikash Kumar, Sergey Levine and Chelsea
Finn, [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
(2023)](https://arxiv.org/abs/2304.13705). This study adapts the algorithm to
numerical simulator observations; it does not claim to reproduce the original
ACT system. See [the ACT report](docs/act.md) for implementation details.
