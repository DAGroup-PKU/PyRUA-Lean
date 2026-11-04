# Notices

The DataFarm source code in this repository is released under the MIT License
(`LICENSE`). Components that derive from third-party projects keep their upstream licenses, listed below. Nothing in this repository grants rights to models,
simulators, services or data that it does not contain.

## Third-party code in this repository

| Path | Upstream | License | Notes |
|---|---|---|---|
| `vendor/trainer/` | [openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence) | Apache-2.0 (`vendor/trainer/LICENSE`) | Modified by the DataFarm authors (Apache-2.0 section 4(b)): dataset adapters, joint/EEF action modes, admission-bound training entry points; see `vendor/trainer/README.md`. Only the files that this project trains with are included, and each modified file carries a notice. `LICENSE_GEMMA.txt` applies to PaliGemma weights, which are not included. |
| `vendor/videox_fun/` | [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) | Apache-2.0 | A patch against upstream for the control pipeline and edge (HED) control preparation; see the README in that directory. |
| `src/datafarm/render/h3/h3_lora.py` | Sol-H3 `h3_runtime/lora.py` | Not yet confirmed | Unmodified copy (sha256 `0e8222b76e086f8f100092e325a08e86bfc8bd33fb404c8b8b0cdd4d3d56362c`). Not covered by the MIT License of this repository until its terms are confirmed. |


## Not included, obtained separately

| Component | Used for | Terms |
|---|---|---|
| MiniMax-H3 weights | default renderer (`docs/reference/rendering.md`) | Obtain from their publishers. The MiniMax-H3 model license restricts some uses of generated outputs (including training other AI models) and some territories. Review it before rendering training data with H3; this repository grants no rights to the model or its outputs. |
| Seedance API | second renderer | Hosted service under its provider's terms. |
| USDCraft | optional object-asset generation (`docs/reference/usdcraft.md`) | Its own license; install it separately. |
| RoboTwin 2.0, SAPIEN, Isaac Sim | simulation | Their own licenses. |
| GPU scheduler | job placement (`docs/deployment.md`) | Supplied by the deployment. |
| PiPER robot description and SDK | robot model, joint limits | From the robot vendor. |
| pi0.5 / PaliGemma checkpoints | policy training | openpi and Gemma terms. |
| Real calibration episodes, rendered datasets, checkpoints | data | Not distributed with the code. |

Run records, receipts and certificates produced by this code bind content by hash; they do
not grant redistribution rights for the content they describe.
