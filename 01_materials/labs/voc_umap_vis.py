#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as etree
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np
import plotly.graph_objs as go
import umap
from plotly.colors import qualitative
from plotly.io import to_html


FILTERS = {"dog", "cat", "bus", "car", "aeroplane"}


def load_filtered_annotations(annotation_folder: Union[str, Path]) -> List[Dict[str, str]]:
    annotations: List[Dict[str, str]] = []
    for xml_path in sorted(Path(annotation_folder).glob("*.xml")):
        tree = etree.parse(xml_path)
        matches = [
            obj for obj in tree.findall("./object")
            if (obj.findtext("name") or "") in FILTERS
        ]
        if len(matches) != 1:
            continue

        filename = tree.findtext("./filename")
        label = matches[0].findtext("name")
        if not filename or not label:
            continue

        annotations.append({"filename": filename, "class": label})
    return annotations


def make_label_colors(labels: List[str]) -> Dict[str, str]:
    colors: Dict[str, str] = {}
    palette = cycle(qualitative.Plotly)
    for lab in dict.fromkeys(labels):
        colors[lab] = next(palette)
    return colors


def pooled_features(reprs: np.ndarray, method: str = "avg") -> np.ndarray:
    # Expect reprs shape: (N, H, W, C)
    if reprs.ndim != 4:
        raise ValueError(f"Expected 4D reprs (N,H,W,C), got {reprs.shape}")
    if method == "avg":
        return reprs.mean(axis=(1, 2))
    elif method == "max":
        return reprs.max(axis=(1, 2))
    elif method == "flatten":
        return reprs.reshape(reprs.shape[0], -1)
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def _group_indices(labels: List[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(lab, []).append(idx)
    return groups


def build_traces_2d(emb2d: np.ndarray, labels: List[str], filenames: List[str], colors: Dict[str, str]):
    traces = []
    for lab, idx in _group_indices(labels).items():
        traces.append(go.Scattergl(
            x=emb2d[idx, 0],
            y=emb2d[idx, 1],
            mode="markers",
            name=lab,
            marker=dict(size=6, opacity=0.85, color=colors[lab]),
            text=[f"{filenames[i]} | {labels[i]}" for i in idx],
            hoverinfo="text"
        ))
    return traces


def build_traces_3d(emb3d: np.ndarray, labels: List[str], filenames: List[str], colors: Dict[str, str]):
    traces = []
    for lab, idx in _group_indices(labels).items():
        traces.append(go.Scatter3d(
            x=emb3d[idx, 0],
            y=emb3d[idx, 1],
            z=emb3d[idx, 2],
            mode="markers",
            name=lab,
            marker=dict(size=3, opacity=0.85, color=colors[lab]),
            text=[f"{filenames[i]} | {labels[i]}" for i in idx],
            hoverinfo="text"
        ))
    return traces


def render_html(fig2d, fig3d, output_html: Path):
    sections = [
        ("2D", fig2d),
        ("3D", fig3d),
    ]
    fragments = []
    for idx, (title, fig) in enumerate(sections):
        figure_html = to_html(fig, include_plotlyjs="cdn" if idx == 0 else False, full_html=False)
        fragments.append(f'<section class="plot"><h2>{title}</h2>{figure_html}</section>')
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VOC UMAP Embeddings</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
h2 {{ margin: 24px 0 8px 0; font-size: 18px; }}
.note {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
.plot {{ margin: 16px 0; }}
</style>
</head>
<body>
<div class="container">
  <h1>VOC UMAP Embeddings</h1>
  <div class="note">Interactive scatter plots of precomputed ResNet50 representations reduced with UMAP. Hover to see filename and label.</div>
  {''.join(fragments)}
</div>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    print(f"Wrote {output_html}")


def main():
    parser = argparse.ArgumentParser(description="UMAP visualization of VOC embeddings")
    parser.add_argument("--cache", default="voc_representations.h5", help="Path to H5 with dataset 'reprs'")
    parser.add_argument("--ann_dir", default="VOCdevkit/VOC2007/Annotations", help="VOC Annotations folder")
    parser.add_argument("--pool", default="avg", choices=["avg", "max", "flatten"], help="Pooling for features")
    parser.add_argument("--metric", default="cosine", help="UMAP metric")
    parser.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, dest="min_dist", help="UMAP min_dist")
    parser.add_argument("--output", default="voc_umap.html", help="Output HTML file")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    ann_dir = Path(args.ann_dir)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation folder not found: {ann_dir}")

    # Load representations
    with h5py.File(cache_path, "r") as h5f:
        if "reprs" not in h5f:
            raise KeyError("Dataset 'reprs' not found in H5 file")
        reprs = h5f["reprs"][:]

    # Rebuild annotations to match order used in notebook
    annotations = load_filtered_annotations(ann_dir)
    if len(annotations) != len(reprs):
        raise RuntimeError(f"Count mismatch: {len(reprs)} reprs vs {len(annotations)} annotations")

    labels = [ann["class"] for ann in annotations]
    filenames = [ann["filename"] for ann in annotations]

    # Pool features and run UMAP
    feats = pooled_features(reprs, method=args.pool)

    reducer2d = umap.UMAP(
        n_components=2, n_neighbors=args.neighbors, min_dist=args.min_dist,
        metric=args.metric, random_state=42
    )
    emb2d = reducer2d.fit_transform(feats)

    reducer3d = umap.UMAP(
        n_components=3, n_neighbors=args.neighbors, min_dist=args.min_dist,
        metric=args.metric, random_state=42
    )
    emb3d = reducer3d.fit_transform(feats)

    # Colors and figures
    colors = make_label_colors(labels)

    fig2d = go.Figure(data=build_traces_2d(emb2d, labels, filenames, colors))
    fig2d.update_layout(
        title="UMAP 2D (metric: {}, neighbors: {}, min_dist: {})".format(args.metric, args.neighbors, args.min_dist),
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        legend_title="Class", template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        height=700
    )

    fig3d = go.Figure(data=build_traces_3d(emb3d, labels, filenames, colors))
    fig3d.update_layout(
        title="UMAP 3D (metric: {}, neighbors: {}, min_dist: {})".format(args.metric, args.neighbors, args.min_dist),
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        legend_title="Class", template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        height=800
    )

    render_html(fig2d, fig3d, Path(args.output))


if __name__ == "__main__":
    main()
