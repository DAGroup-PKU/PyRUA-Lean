import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class PiperEefXyz3dInputs(transforms.DataTransformFn):
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape[-1] != 14:
            raise ValueError(f"Expected a 14D XYZ3D state, got shape {state.shape}")

        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(f"Unexpected XYZ3D camera names: {tuple(sorted(unexpected))}")

        def convert_image(image: np.ndarray) -> np.ndarray:
            image = np.asarray(image)
            if np.issubdtype(image.dtype, np.floating):
                image = (255 * image).astype(np.uint8)
            return einops.rearrange(image, "c h w -> h w c")

        base_image = convert_image(in_images["cam_high"])
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for destination, source in {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }.items():
            if source in in_images:
                images[destination] = convert_image(in_images[source])
                image_masks[destination] = np.True_
            else:
                images[destination] = np.zeros_like(base_image)
                image_masks[destination] = np.False_

        result = {"image": images, "image_mask": image_masks, "state": state.copy()}
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != 14:
                raise ValueError(f"Expected 14D XYZ3D actions, got shape {actions.shape}")
            result["actions"] = actions.copy()
        for key in ("prompt", "episode_index", "frame_index"):
            if key in data:
                result[key] = data[key]
        return result


@dataclasses.dataclass(frozen=True)
class PiperEefXyz3dOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
