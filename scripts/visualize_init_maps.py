#!/usr/bin/env python3
"""Visualize TagMap + ORB init map exported by TagMapExporter (interactive 3D).

Expected files under export_dir (default: Examples/TagFusion/XREAL_Glass/tag_export):
  - tag_map_corners.csv
  - orb_init_map_points.csv
  - tag_init_keyframes_tum.txt   (optional)
  - orb_init_keyframes_tum.txt   (optional)

Outputs (by default in export_dir):
  - init_maps_3d.html   interactive 3D (Plotly, rotate/zoom/pan)
  - init_maps_3d.ply    combined point cloud (optional, --ply)
  - init_maps_vis.png   static snapshot (optional, --png)

Usage:
  python3 scripts/visualize_init_maps.py
  python3 scripts/visualize_init_maps.py --export-dir path/to/tag_export
  python3 scripts/visualize_init_maps.py --html out.html --ply out.ply --png out.png
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
    corners: np.ndarray  # (4, 3) or empty


@dataclass
class InitMapData:
    tags: list[TagRecord]
    orb_pts: np.ndarray
    tag_traj: np.ndarray
    orb_traj: np.ndarray


def load_tag_corners(path: Path) -> list[TagRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"missing tag map: {path}")

    tags: list[TagRecord] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row["tag_id"])
            is_fixed = int(row.get("is_fixed", "0")) != 0
            has_corners = int(row.get("has_corners", "0")) != 0
            if not has_corners:
                tags.append(TagRecord(tag_id, is_fixed, np.empty((0, 3))))
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
            tags.append(TagRecord(tag_id, is_fixed, corners))
    return tags


def load_orb_points(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing ORB map points: {path}")

    pts: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append([float(row["x"]), float(row["y"]), float(row["z"])])
    if not pts:
        return np.empty((0, 3))
    return np.asarray(pts, dtype=np.float64)


def load_tum_trajectory(path: Path) -> np.ndarray:
    """Return Nx3 camera centers (TUM: timestamp tx ty tz qx qy qz qw)."""
    if not path.is_file():
        return np.empty((0, 3))

    centers: list[list[float]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            centers.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not centers:
        return np.empty((0, 3))
    return np.asarray(centers, dtype=np.float64)


def load_init_maps(export_dir: Path) -> InitMapData:
    return InitMapData(
        tags=load_tag_corners(export_dir / "tag_map_corners.csv"),
        orb_pts=load_orb_points(export_dir / "orb_init_map_points.csv"),
        tag_traj=load_tum_trajectory(export_dir / "tag_init_keyframes_tum.txt"),
        orb_traj=load_tum_trajectory(export_dir / "orb_init_keyframes_tum.txt"),
    )


def combined_points(data: InitMapData) -> np.ndarray:
    chunks: list[np.ndarray] = []
    if data.orb_pts.size:
        chunks.append(data.orb_pts)
    tag_corners = [t.corners for t in data.tags if t.corners.size]
    if tag_corners:
        chunks.append(np.vstack(tag_corners))
    if data.tag_traj.size:
        chunks.append(data.tag_traj)
    if data.orb_traj.size:
        chunks.append(data.orb_traj)
    if not chunks:
        return np.empty((0, 3))
    return np.vstack(chunks)


def axis_ranges(points: np.ndarray, margin: float = 0.08) -> dict[str, list[float]]:
    if points.size == 0:
        return {"x": [-1, 1], "y": [-1, 1], "z": [-1, 1]}
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


def tag_color(tag_id: int) -> str:
    return TAG_COLORS[tag_id % len(TAG_COLORS)]


def build_plotly_figure(data: InitMapData, export_dir: Path) -> go.Figure:
    fig = go.Figure()
    combined = combined_points(data)

    if data.orb_pts.size:
        fig.add_trace(
            go.Scatter3d(
                x=data.orb_pts[:, 0],
                y=data.orb_pts[:, 1],
                z=data.orb_pts[:, 2],
                mode="markers",
                name=f"ORB map points ({data.orb_pts.shape[0]})",
                marker=dict(size=3, color="#4C72B0", opacity=0.65),
                hovertemplate="ORB<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )

    for tag in data.tags:
        if tag.corners.shape != (4, 3):
            continue
        c = tag.corners
        color = tag_color(tag.tag_id)
        closed = np.vstack([c, c[0:1]])

        fig.add_trace(
            go.Scatter3d(
                x=closed[:, 0],
                y=closed[:, 1],
                z=closed[:, 2],
                mode="lines",
                name=f"Tag {tag.tag_id}",
                line=dict(color=color, width=6),
                hovertemplate=f"Tag {tag.tag_id}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Mesh3d(
                x=c[:, 0],
                y=c[:, 1],
                z=c[:, 2],
                i=[0],
                j=[1],
                k=[2],
                opacity=0.35,
                color=color,
                showscale=False,
                hoverinfo="skip",
                name=f"Tag {tag.tag_id} face",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Mesh3d(
                x=c[:, 0],
                y=c[:, 1],
                z=c[:, 2],
                i=[0],
                j=[2],
                k=[3],
                opacity=0.35,
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
                textfont=dict(size=14, color="black"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if data.tag_traj.size:
        fig.add_trace(
            go.Scatter3d(
                x=data.tag_traj[:, 0],
                y=data.tag_traj[:, 1],
                z=data.tag_traj[:, 2],
                mode="lines+markers",
                name=f"Tag init KF ({data.tag_traj.shape[0]})",
                line=dict(color="#2CA02C", width=5),
                marker=dict(size=5, color="#2CA02C", symbol="circle"),
                hovertemplate="Tag KF<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )

    if data.orb_traj.size:
        fig.add_trace(
            go.Scatter3d(
                x=data.orb_traj[:, 0],
                y=data.orb_traj[:, 1],
                z=data.orb_traj[:, 2],
                mode="lines+markers",
                name=f"ORB init KF ({data.orb_traj.shape[0]})",
                line=dict(color="#D62728", width=4, dash="dash"),
                marker=dict(size=5, color="#D62728", symbol="square"),
                hovertemplate="ORB KF<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )

    n_tag_corners = sum(1 for t in data.tags if t.corners.size)
    ranges = axis_ranges(combined)
    fig.update_layout(
        title=(
            "Init map: TagMap + ORB (same world frame)<br>"
            f"<sup>{export_dir}<br>"
            f"tags={len(data.tags)} (with corners={n_tag_corners}), "
            f"orb_points={data.orb_pts.shape[0]}</sup>"
        ),
        scene=dict(
            xaxis=dict(title="X [m]", range=ranges["x"]),
            yaxis=dict(title="Y [m]", range=ranges["y"]),
            zaxis=dict(title="Z [m]", range=ranges["z"]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.0)),
        ),
        legend=dict(x=0.01, y=0.98),
        margin=dict(l=0, r=0, t=80, b=0),
        width=1100,
        height=800,
    )
    return fig


def save_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path),
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(displayModeBar=True, scrollZoom=True),
    )
    print(f"saved interactive 3D -> {path}")


def save_png(fig: go.Figure, path: Path) -> None:
    try:
        import plotly.io as pio

        path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_image(fig, str(path), width=1100, height=800, scale=2)
        print(f"saved PNG snapshot -> {path}")
    except Exception as exc:
        print(f"PNG export skipped ({exc}); install kaleido: pip install kaleido")


def save_ply(data: InitMapData, path: Path) -> None:
    """Write ASCII PLY point cloud: ORB points + tag corners + KF centers."""
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices: list[tuple[float, float, float, int, int, int]] = []

    for p in data.orb_pts:
        vertices.append((p[0], p[1], p[2], 76, 114, 176))

    for tag in data.tags:
        if tag.corners.shape != (4, 3):
            continue
        r, g, b = _hex_to_rgb(tag_color(tag.tag_id))
        for c in tag.corners:
            vertices.append((c[0], c[1], c[2], r, g, b))

    for p in data.tag_traj:
        vertices.append((p[0], p[1], p[2], 44, 160, 44))

    for p in data.orb_traj:
        vertices.append((p[0], p[1], p[2], 214, 39, 40))

    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in vertices:
            f.write(f"{x:.9f} {y:.9f} {z:.9f} {r} {g} {b}\n")

    print(f"saved PLY point cloud -> {path} (vertices={len(vertices)})")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def visualize_3d(
    export_dir: Path,
    html_path: Path,
    png_path: Path | None,
    ply_path: Path | None,
    open_browser: bool,
) -> None:
    data = load_init_maps(export_dir)
    fig = build_plotly_figure(data, export_dir)

    save_html(fig, html_path)
    if png_path is not None:
        save_png(fig, png_path)
    if ply_path is not None:
        save_ply(data, ply_path)

    if open_browser:
        import webbrowser

        webbrowser.open(html_path.as_uri())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interactive 3D visualization for TagMap + ORB init maps."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help=f"input export directory (default: {DEFAULT_EXPORT_DIR})",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="output HTML path (default: <export-dir>/init_maps_3d.html)",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="optional static PNG snapshot",
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="optional PLY point cloud export",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open HTML in browser after generation",
    )
    parser.add_argument(
        "--no-ply",
        action="store_true",
        help="skip default PLY export",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    if not export_dir.is_dir():
        print(f"export dir not found: {export_dir}")
        return 1

    html_path = (args.html or (export_dir / "init_maps_3d.html")).resolve()
    png_path = args.png.resolve() if args.png else None
    if args.no_ply:
        ply_path = None
    elif args.ply is not None:
        ply_path = args.ply.resolve()
    else:
        ply_path = (export_dir / "init_maps_3d.ply").resolve()

    try:
        visualize_3d(
            export_dir=export_dir,
            html_path=html_path,
            png_path=png_path,
            ply_path=ply_path,
            open_browser=args.open,
        )
    except FileNotFoundError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
