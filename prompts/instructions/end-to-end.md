# From request to training

DataFarm's output is training data with provenance, action labels and review evidence, plus policy results when requested. The
system has five versioned intermediate representations, in order: Grounding IR (real basis), Task IR (task and expert), Episode IR
(authoritative trajectory), Render IR (rendering output), Admission Certificate. The names only indicate duties; validity depends on
the input, the evidence and an independent verifier.

1. **Parse the request:** task, sample target, data mode, real references, allowed changes, training state mode, budget. If the mode
   is unclear, ask first. "Randomized table height" does not mean "freely replacing the real scene".
2. **Real basis and task:** save the camera, robot, object and reference sources. The AI-written expert is developed, rolled out and
   held out in isolation through the expert harness. A candidate real-camera calibration without acceptance is not a general hardware calibration.
3. **Collect trajectories:** authoritative simulated replay, saving the same-source actions, state and images, seeds, versions and actual
   joint readbacks. New collection and replanning obey the joint-margin requirements. Never crop only the labels while keeping the images.
4. **Small-batch rendering:** freeze the intent, prompt and real references, verify the actual payload, and review the complete action
   stages and the real appearance. Fix systemic failures first; do not expand to the full batch.
5. **Production and review:** verify each item as it completes, independently recheck rejections first, retry a bounded number of
   times, and keep the fixed candidate denominator and replacement relationships. Full count does not replace full quality.
6. **Data admission:** issue the certificate once the independent motion/topology and style/reference evidence is complete. The builder
   checks video/action synchronization and the source hashes.
7. **Training and evaluation:** an explicit state mode; register the job, checkpoint, evaluation protocol and metrics. Started does not
   mean complete, and complete does not mean deployment succeeded.
8. **Close-out:** report what is completed, missing, failed or pending verification for the stage range the user asked for, keeping
   rejections and costs. The ledger closes on verifiable outputs; a hand-written all-pass is not allowed.

The new-task entry first executes `new-task-adaptation` and `sim-grounding`: the orchestrator generates the adapter, judge, task
requirements and render overlay; the user does not fill them in. Documents and CPU contract tests are not evidence of real task success.
