#!/usr/bin/env python3
"""Visualize the final TagMap (all MapTags with world corners).

Reads tag_map_corners.csv from TagMapExporter (online overwrite + final dump).
Does not overlay ORB-init points or init keyframes.

Usage:
  python3 scripts/visualize_tag_map.py
  python3 scripts/visualize_tag_map.py --export-dir path/to/tag_export
  python3 scripts/visualize_tag_map.py --csv path/to/tag_map_corners.csv --html out.html
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


SCRIPT_DIR = Path(__file__).resolve().parent
ORB_ROOT = SCRIPT_DIR.parent
DEFAULT_EXPORT_DIR = ORB_ROOT / "Examples/TagFusion/XREAL_Glass/tag_export"

TAG_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


@dataclass
class TagRecord:
    tag_id: int
    is_fixed: bool
    corners: np.ndarray  # (4, 3)


def load_tag_corners(path: Path) -> tuple[list[TagRecord], list[int]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing tag map: {path}")

    posed: list[TagRecord] = []
    no_pose: list[int] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row["tag_id"])
            is_fixed = int(row.get("is_fixed", "0")) != 0
            has_corners = int(row.get("has_corners", "0")) != 0
            if not has_corners:
                no_pose.append(tag_id)
                continue
            corners = np.array(
                [
                    [float(row["c0_x"]), float(row["c0_y"]), float(row["c0_z"])],
                    [float(row["c1_x"]), float(row["c1_y"]), float(row["c1_z"])],
                    [float(row["c2_x"]), float(row["c2_y"]), float(row["c2_z"])],
                    [float(row["c3_x"]), float(row["c3_y"]), float(row["c3_z"])],
                ],
                dtype=np.float64,
            )
            posed.append(TagRecord(tag_id, is_fixed, corners))
    posed.sort(key=lambda t: t.tag_id)
    return posed, no_pose


def load_tum_trajectory(path: Path, stride: int = 5) -> np.ndarray:
    if not path.is_file():
        return np.empty((0, 3))
    centers: list[list[float]] = []
    with path.open() as f:
        n = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            if n % stride == 0:
                centers.append([float(parts[1]), float(parts[2]), float(parts[3])])
            n += 1
    if not centers:
        return np.empty((0, 3))
    return np.asarray(centers, dtype=np.float64)


def tag_color(tag_id: int) -> str:
    return TAG_COLORS[tag_id % len(TAG_COLORS)]


def axis_ranges(points: np.ndarray, margin: float = 0.12) -> dict[str, list[float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius <= 0.0:
        radius = 0.1
    radius *= 1.0 + margin
    return {
        "x": [center[0] - radius, center[0] + radius],
        "y": [center[1] - radius, center[1] + radius],
        "z": [center[2] - radius, center[2] + radius],
    }


def build_figure(
    tags: list[TagRecord],
    no_pose: list[int],
    cam_traj: np.ndarray,
    source: Path,
) -> go.Figure:
    fig = go.Figure()
    chunks = [t.corners for t in tags]
    if cam_traj.size:
        chunks.append(cam_traj)
    combined = np.vstack(chunks) if chunks else np.zeros((1, 3))

    if cam_traj.size:
        fig.add_trace(
            go.Scatter3d(
                x=cam_traj[:, 0],
                y=cam_traj[:, 1],
                z=cam_traj[:, 2],
                mode="lines",
                name=f"camera traj ({cam_traj.shape[0]})",
                line=dict(color="#bbbbbb", width=3),
                hovertemplate="cam<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )

    for tag in tags:
        c = tag.corners
        color = tag_color(tag.tag_id)
        closed = np.vstack([c, c[0:1]])
        suffix = " FIXED" if tag.is_fixed else ""
        fig.add_trace(
            go.Scatter3d(
                x=closed[:, 0],
                y=closed[:, 1],
                z=closed[:, 2],
                mode="lines",
                name=f"Tag {tag.tag_id}{suffix}",
                line=dict(color=color, width=6),
                hovertemplate=(
                    f"Tag {tag.tag_id}{suffix}"
                    "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Mesh3d(
                x=c[:, 0],
                y=c[:, 1],
                z=c[:, 2],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                opacity=0.40,
                color=color,
                showscale=False,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        center = c.mean(axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="text",
                text=[str(tag.tag_id)],
                textfont=dict(size=13, color="black"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    no_pose_txt = ", ".join(str(i) for i in no_pose) if no_pose else "none"
    fig.update_layout(
        title=(
            "Final TagMap (world corners)<br>"
            f"<sup>{source}<br>"
            f"posed={len(tags)}, no_pose={len(no_pose)} [{no_pose_txt}]</sup>"
        ),
        scene=dict(
            xaxis=dict(title="X [m]", range=axis_ranges(combined)["x"]),
            yaxis=dict(title="Y [m]", range=axis_ranges(combined)["y"]),
            zaxis=dict(title="Z [m]", range=axis_ranges(combined)["z"]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.0)),
        ),
        legend=dict(x=0.01, y=0.99, font=dict(size=11)),
        margin=dict(l=0, r=0, t=90, b=0),
        width=1100,
        height=800,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive 3D of the final TagMap (all posed MapTags)."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help=f"export directory (default: {DEFAULT_EXPORT_DIR})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="tag_map_corners.csv (default: <export-dir>/tag_map_corners.csv)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="output HTML (default: <export-dir>/tag_map_3d.html)",
    )
    parser.add_argument(
        "--no-traj",
        action="store_true",
        help="do not overlay tag_camera_trajectory.txt",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open HTML in browser after generation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    csv_path = (args.csv or (export_dir / "tag_map_corners.csv")).resolve()
    html_path = (args.html or (export_dir / "tag_map_3d.html")).resolve()

    tags, no_pose = load_tag_corners(csv_path)
    if not tags:
        print(f"no posed tags in {csv_path}")
        return 1

    cam_traj = np.empty((0, 3))
    traj_path = export_dir / "tag_camera_trajectory.txt"
    if not args.no_traj:
        cam_traj = load_tum_trajectory(traj_path)

    fig = build_figure(tags, no_pose, cam_traj, csv_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(displayModeBar=True, scrollZoom=True),
    )
    print(f"saved final TagMap 3D -> {html_path}")
    print(f"posed tags ({len(tags)}): {[t.tag_id for t in tags]}")
    if no_pose:
        print(f"no world corners ({len(no_pose)}): {no_pose}")

    if args.open:
        import webbrowser

        webbrowser.open(html_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
