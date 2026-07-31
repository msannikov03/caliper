# Real-robot URDF corpus

URDFs vendored verbatim for cross-validation; meshes intentionally not vendored
(colliders drop loudly via `dropped_collider_frames`, visuals keep `path=None`).

- `panda.urdf` — Franka Emika Panda, from [example-robot-data](https://github.com/Gepetto/example-robot-data) (`robots/panda_description/urdf/panda.urdf`), BSD-2-Clause.
- `so101_new_calib.urdf` — SO-101 arm, from [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (`Simulation/SO101/so101_new_calib.urdf`), Apache-2.0.
- `so100.urdf` — SO-100 arm, from [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (`Simulation/SO100/so100.urdf`), Apache-2.0.
- `gen3_lite.urdf` — Kinova Gen3 lite, from [ros2_kortex](https://github.com/Kinovarobotics/ros2_kortex) (`kortex_description/robots/gen3_lite.urdf`), BSD-3-Clause.
