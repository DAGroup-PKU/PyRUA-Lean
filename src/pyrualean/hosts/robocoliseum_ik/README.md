# GenieSim's IK description of the G2

The files in this directory come from [GenieSim](https://github.com/AgibotTech/genie_sim),
`source/geniesim_benchmark/src/geniesim_benchmark/utils/IK-SDK/` at commit
`a2bbec0c0e2d05f7cb5c13c9a75bcbc16ea229be`, and are licensed under the Mozilla Public License 2.0
(`LICENSE` in this directory), not under the Apache License 2.0 of the rest of PyRUA-Lean.

- `G2_NO_GRIPPER.urdf` and `g2_solver.yaml`: unchanged copies.
- `g2_goal.yaml`: a modified copy of `g2_solver.yaml`. The optimizer may take more iterations and time
  (`maxeval` 1000 → 20000, `maxtime` 0.05 → 0.3 s), and the velocity and acceleration costs are off
  (weight 0), so that a whole goal pose is solved at once.

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL
was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
