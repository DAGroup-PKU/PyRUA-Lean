"""Pixel layouts shared by the H3 render tools: the T-layout source raster and the H3 canvas (stdlib only)."""
from __future__ import annotations

from dataclasses import dataclass

# Camera panels of the 1280x1080 T layout: view -> (y0, y1, x0, x1)
T_LAYOUT_PANELS = {"head": (0, 720, 0, 1280), "left_wrist": (720, 1080, 0, 640), "right_wrist": (720, 1080, 640, 1280)}


@dataclass(frozen=True)
class H3SpatialContract:
    """One evidence-visible coordinate system for H3 video.

    The simulator's active three-view raster is 1280x1080 (32:27).  MiniMax H3 renders on a 1024x768 4:3 canvas.  The
    route therefore pads 80 pixels on either side before a uniform 1440x1080 -> 1024x768 resize.  The active image
    occupies x=[57, 967) on the H3 canvas.  Assembly crops that active area and resizes it back to 1280x1080; H3 never
    sees a stretched active raster.
    """

    contract_id: str = "h3-guttered-active-1280x1080-v1"
    active_width: int = 1280
    active_height: int = 1080
    padded_width: int = 1440
    padded_height: int = 1080
    input_gutter: int = 80
    canvas_width: int = 1024
    canvas_height: int = 768
    canvas_active_x0: int = 57
    canvas_active_x1: int = 967

    @property
    def canvas_active_width(self) -> int:
        return self.canvas_active_x1 - self.canvas_active_x0

    @property
    def canvas_head_height(self) -> int:
        return int(round(self.canvas_height * 720 / 1080))

    def canvas_views(self) -> dict[str, tuple[int, int, int, int]]:
        """Camera panels on the H3 canvas: view -> (y0, y1, x0, x1)."""
        x0, x1, hy = self.canvas_active_x0, self.canvas_active_x1, self.canvas_head_height
        xm = (x0 + x1) // 2
        return {"head": (0, hy, x0, x1), "left_wrist": (hy, self.canvas_height, x0, xm),
                "right_wrist": (hy, self.canvas_height, xm, x1)}

    def to_dict(self) -> dict[str, int | str | list[int]]:
        return {
            "contract_id": self.contract_id,
            "source_active_size": [self.active_width, self.active_height],
            "padded_size": [self.padded_width, self.padded_height],
            "input_gutter_px": self.input_gutter,
            "h3_canvas_size": [self.canvas_width, self.canvas_height],
            "canvas_active_x": [self.canvas_active_x0, self.canvas_active_x1],
            "final_output_size": [self.active_width, self.active_height],
            "final_transform": "crop-active-then-lanczos-resize",
        }

    def ffmpeg_to_canvas(self) -> str:
        return (
            f"scale={self.active_width}:{self.active_height}:flags=lanczos,"
            f"pad={self.padded_width}:{self.padded_height}:{self.input_gutter}:0:color=black,"
            f"scale={self.canvas_width}:{self.canvas_height}:flags=lanczos,"
            f"drawbox=x=0:y=0:w={self.canvas_active_x0}:h={self.canvas_height}:"
            "color=black:t=fill,"
            f"drawbox=x={self.canvas_active_x1}:y=0:"
            f"w={self.canvas_width - self.canvas_active_x1}:h={self.canvas_height}:"
            "color=black:t=fill,setsar=1"
        )

    def ffmpeg_canvas_to_active(self) -> str:
        return (
            f"crop={self.canvas_active_width}:{self.canvas_height}:{self.canvas_active_x0}:0,"
            f"scale={self.active_width}:{self.active_height}:flags=lanczos,setsar=1"
        )


SPATIAL = H3SpatialContract()
