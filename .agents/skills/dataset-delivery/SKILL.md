---
name: dataset-delivery
description: Turn independently reviewed, source-bound full episodes into aligned batch datasets and verifiable delivery, with optional requested training.
---

# Definition of done for data production

The inputs are the actions/actual states of complete simulated episodes, the white mesh, the real references, the photorealistic final videos and the independent reviews.
First read docs/reference/delivery-and-admission.md.

1. For each episode, check the hashes of the input, actions, images, frame mapping, prompt, real references and the finally selected media.
   If a video/reference/action was repaired it must be re-reviewed, and the earlier certificate must not be reused. Record continuation-based mixed provenance honestly, and do not claim everything was regenerated.
2. Review separately the structure/task, adoption of the real references, real-world style, camera/timing, and joint data; a machine pass and
   a provider success do not equal final admission. The independent reviewer must personally decode the real references and the generated video and view them side by side,
   save the bound comparison images and frame numbers, and cover the real baseline, the robot body, the attached cameras/cables and objects that actually exist,
   the table/background, and the photographic feel of all views declared by the current task; components that do not exist use not_applicable with evidence and reason,
   and must not be invented. Local textures and text-only reviews cannot count as a style pass.
   OOD must still be a reasonable real change along approved axes. unverified does not count as pass. Real hard shadows are not a defect in themselves; exposure serves only as a coarse safeguard.
3. Fix the target number of qualified episodes and the denominator of all attempts. Infrastructure failures, quality rejections, replacement relations and costs are all kept; when quality
   is not met, re-collect/repair within the authorized budget, and do not delete bad samples and then write 100% success. Run a representative complete canary before scaling.
4. The agent itself generates the data encoding/export adapter for the new task, stating the state/action dimensions, order, units,
   absolute/delta representation, time alignment and camera names; verify by reading the exported samples, not only by checking that the directory is non-empty. Do not use the state schema of another task
   to interpret the new device. For a pi0.5 dataset, `tools/build_pi05_dataset.py` builds the inline-224 LeRobot dataset from the admitted cohort. Record the data manifest, the provenance certificates and a runnable reading example.
5. Issue admission independently and verify it at the actual build entry point; use delivery.json to record the admitted episodes,
   all attempt manifests, the dataset manifest, the grounding record, the input/instruction versions and the independent reviews.
   For tasks run by pipeline_agent drive/supervise, the episode review must be completed through the state.review_requests queue;
   the review file writes runtime_review=<request ID> and lists it in the evidence of that review report; hand-written execution records are rejected.
   Run pipeline_agent verify.
6. Only if the request includes training, prepare an explicit state_conditioning, the action range/normalization and deployment consistency, launch through
   an admission-bound tool, and check the actual update count/checkpoint/inference before reporting. Data delivery does not equal success on the real robot.

The delivery note must let a new reader answer what the input is, how much data there is, how to read it, why it can be trusted, and what is unverified.

When the user explicitly wants only white-mesh output, use `--stop-after white` in prepare to fix the boundary, and produce the white-mesh delivery according to
`docs/reference/delivery-and-admission.md`; do not submit rendering, and do not forge adoption of references, photographic style or training admission.
The white mesh still needs independent evidence for the full task, actions, joints, clocks and all declared views. In all other cases the default full-delivery rules apply.
