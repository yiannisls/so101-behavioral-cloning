# Demonstrations

Demonstrations are local files, excluded from Git. The initial collection has 22
episodes. Episode 13 did not release the cube and was excluded. The cleaned copy
contains 21 episodes and 11,894 observations. Empty sessions and the earlier
35-frame practice episode are not used. Original recordings remain intact.

The recording stores raw observations and gripper control commands. Preparing
XYZ actions produces one fewer transition per episode, because the final
observation has no next observation. The processed baseline therefore contains
11,873 transitions. A later seven-episode collection contains 3,721 observations
and 3,714 processed transitions. See the experiment log for selection and results.

See `../configs/data_selection.json` for paths and original episode numbers.
In the cleaned store, episodes are renumbered consecutively; its
`source_episode_numbers` attribute preserves the original numbers.

The cleaned demonstrations have been copied locally from the coursework folder.
They are not downloaded when someone clones this repository. Other users can
record their own demonstrations with `scripts/record_teleop_demos.py`.
