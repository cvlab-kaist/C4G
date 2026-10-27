"""Per-view supervision maps that travel with the images through the shims.

All maps are unbatched ``[view, h, w]`` (``[view, h, w, 2]`` for the flow) and
live in the same pixel grid as ``views["image"]`` at every stage, so the crop
shim and the flip augmentation must transform them together with the images.

    depth            metric-ish depth of each view in pose units, 0 = invalid
    motion_flow      2D flow (dx, dy) in pixels from this view to the next
                     timestamp of the same camera
    motion_depth_next  depth (in the next camera frame) of the pixel's
                     correspondence at the next timestamp, 0 = invalid
    motion_mask      1 = moving pixel, 0 = static, -1 = unknown
    motion_valid     1 where flow / depth_next are trustworthy
    mask             legacy dynamic mask consumed by LossDynamicMask
"""

SCALAR_MAP_KEYS = ("depth", "motion_depth_next", "motion_mask", "motion_valid", "mask")
FLOW_MAP_KEYS = ("motion_flow",)
MOTION_MAP_KEYS = SCALAR_MAP_KEYS + FLOW_MAP_KEYS
