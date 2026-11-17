# Quickstart sample: stack paper cups

One real episode and a saved user request, for trying [Run a task](../../docs/quickstart.md#run-a-task).

The robot is an AgileX Cobot Magic with two Piper arms. It stacks the paper cups on the table into a single stack, then returns both arms to their starting pose. This is the input episode of the stack-cups task in the paper.

```
episode/
  cam_high.mp4            head camera, 1280x720, 30 fps, 673 frames (22.4 s)
  cam_left_wrist.mp4      left wrist camera, same timing
  cam_right_wrist.mp4     right wrist camera, same timing
  puppet_state.json       joint and end-effector states of both arms
  camera_parameters.json  camera intrinsics
  quality.json            the recorder's decode check
request.txt               the user's request
request.zh.txt            the same request in Chinese
SHA256SUMS
```

From the repository root:

```bash
python tools/pipeline_agent.py prepare --episode examples/quickstart/episode \
  --request-file examples/quickstart/request.txt --target-count 5 --stop-after white --output /data/task
```

Pass the whole `episode/` directory: `prepare` pins every file in it by sha256. The request asks for 5 independently reviewed simulated episodes and stops before photoreal rendering, which matches `--target-count 5 --stop-after white`.

To keep the repository small, the recorder's side-by-side video of the three views is not included. `quality.json` still lists it, along with other paths from the recording machine.
