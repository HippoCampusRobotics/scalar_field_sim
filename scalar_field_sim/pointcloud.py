"""Helpers for converting scalar field values into PointCloud2 messages.

This module builds RViz-friendly point clouds for:
- ground-truth field visualization,
- belief mean visualization,
- belief variance visualization.

Scalar values are mapped to explicit RGB colors before publishing, so the
colormap does not depend on RViz's own intensity-color settings.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def _values_to_rgb_uint32(
    values: np.ndarray,
    color_min: float,
    color_max: float,
    cmap_name: str = 'viridis',
) -> np.ndarray:
    """Map scalar values to packed 24-bit RGB colors.

    Parameters
    ----------
    values
        Scalar values of shape `(N,)`.
    color_min
        Lower bound of the fixed color range.
    color_max
        Upper bound of the fixed color range.
    cmap_name
        Name of the matplotlib colormap used for color mapping.

    Returns
    -------
    rgb_u32
        Packed RGB values as `uint32`, shape `(N,)`, with layout `0xRRGGBB`.

    Notes
    -----
    Values below `color_min` are clipped to the low end of the colormap.
    Values above `color_max` are clipped to the high end.
    """
    values = np.asarray(values, dtype=np.float32)

    if color_max <= color_min:
        raise ValueError('color_max must be greater than color_min.')

    normalized = (values - color_min) / (color_max - color_min)
    normalized = np.clip(normalized, 0.0, 1.0)

    cmap = colormaps[cmap_name]
    rgba = cmap(normalized)  # shape (N, 4), floats in [0, 1]

    rgb_u8 = (255.0 * rgba[:, :3]).astype(np.uint8)
    rgb_u32 = (
        (rgb_u8[:, 0].astype(np.uint32) << 16)
        | (rgb_u8[:, 1].astype(np.uint32) << 8)
        | rgb_u8[:, 2].astype(np.uint32)
    )
    return rgb_u32


def make_field_pointcloud2(
    positions_xy: np.ndarray,
    values: np.ndarray,
    frame_id: str,
    stamp,
    z_mode: str = 'flat',
    z_offset: float = -0.6,
    height_scale: float = 1.0,
    colormap_min: float = 0.0,
    colormap_max: float = 0.015,
    cmap_name: str = 'viridis',
) -> PointCloud2:
    """Build a PointCloud2 message for scalar field visualization.

    Parameters
    ----------
    positions_xy
        2D point positions of shape `(N, 2)`.
    values
        Scalar values of shape `(N,)`.
    frame_id
        ROS frame of the point cloud.
    stamp
        ROS timestamp for the cloud header.
    z_mode
        Vertical visualization mode:
        - `"flat"`: all points lie on one plane - this is the standard!
        - `"height"`: values are additionally shown as height, for debugging
    z_offset
        Constant vertical offset added to all points.
    height_scale
        Maximum height used when `z_mode == "height"`.
    colormap_min
        Lower bound of the fixed color range.
    colormap_max
        Upper bound of the fixed color range.
    cmap_name
        Name of the matplotlib colormap used for RGB mapping.

    Returns
    -------
    PointCloud2
        Point cloud with fields `x`, `y`, `z`, and `rgb`.

    Notes
    -----
    The `"height"` mode is meant for debugging.
    In `"height"` mode, z values are normalized per cloud to the interval
    `[0, height_scale]` before `z_offset` is added. This makes the vertical
    shape easier to see, but it also means height is relative within one cloud,
    not directly comparable across different clouds.

    Colors, in contrast, use the fixed range `[colormap_min, colormap_max]`.
    """
    positions_xy = np.asarray(positions_xy, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if positions_xy.ndim != 2 or positions_xy.shape[1] != 2:
        raise ValueError('positions_xy must have shape (N, 2).')
    if len(positions_xy) != len(values):
        raise ValueError('positions_xy and values must have matching length.')

    if z_mode == 'flat':
        z = np.zeros(len(values), dtype=np.float32)
    elif z_mode == 'height':
        vmin = float(values.min())
        vmax = float(values.max())
        denom = max(vmax - vmin, 1e-12)
        z = ((values - vmin) / denom * height_scale).astype(np.float32)
    else:
        raise ValueError("z_mode must be 'flat' or 'height'.")

    rgb = _values_to_rgb_uint32(
        values=values,
        color_min=colormap_min,
        color_max=colormap_max,
        cmap_name=cmap_name,
    )

    # RViz expects the rgb field packed into one FLOAT32 field. The packed
    # integer colors are therefore reinterpreted as float32 without changing
    # the underlying bit pattern.
    points = np.column_stack(
        [
            positions_xy[:, 0],
            positions_xy[:, 1],
            z + z_offset,
            rgb.view(np.float32),
        ]
    ).astype(np.float32, copy=False)

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        # PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return point_cloud2.create_cloud(header, fields, points)
