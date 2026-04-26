"""Build a class co-occurrence and boundary-touch graph prior for BiSeNet-RTB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.foodseg_manifest import FoodSegManifestDataset


def parse_args() -> argparse.Namespace:
    """Parse graph-prior builder arguments."""
    parser = argparse.ArgumentParser("Build RTB graph prior from a FoodSeg manifest.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out", type=str, default="datasets/processed/metadata/graph_prior_104.pt")
    parser.add_argument("--stage", type=str, default="full")
    parser.add_argument("--num-classes", type=int, default=104)
    parser.add_argument("--background-id", type=int, default=103)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--dilate", type=int, default=3)
    parser.add_argument("--lambda-smooth", type=float, default=1.0)
    parser.add_argument("--cooc-weight", type=float, default=0.6)
    parser.add_argument("--touch-weight", type=float, default=0.4)
    return parser.parse_args()


def row_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize rows while avoiding divide-by-zero."""
    return x / np.maximum(x.sum(axis=1, keepdims=True), 1e-8)


def main() -> None:
    """Build and save the RTB graph prior artifact."""
    args = parse_args()

    ds = FoodSegManifestDataset(
        manifest_csv=args.manifest,
        data_root=args.data_root,
        train_stage=args.stage,
        transform=None,
    )

    k = args.num_classes
    bg = args.background_id
    ignore = args.ignore_index

    n_i = np.zeros(k, dtype=np.float64)
    n_ij = np.zeros((k, k), dtype=np.float64)
    touch = np.zeros((k, k), dtype=np.float64)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.dilate, args.dilate),
    )

    for _, row in tqdm(ds.df.iterrows(), total=len(ds.df), desc="Build RTB graph prior"):
        mask_path = ds._resolve_existing_path(row=row, key="mask_path", kind="mask")
        mask = np.array(Image.open(mask_path), dtype=np.int64)

        valid = (mask != ignore) & (mask >= 0) & (mask < k)
        labels = np.unique(mask[valid])
        labels = labels[labels != bg]

        if labels.size == 0:
            continue

        n_i[labels] += 1.0

        for c in labels:
            n_ij[c, labels] += 1.0

        class_masks = {}
        for c in labels:
            cm = (mask == c).astype(np.uint8)
            if cm.sum() > 0:
                class_masks[int(c)] = cm

        label_list = list(class_masks.keys())
        dilated = {
            c: cv2.dilate(class_masks[c], kernel, iterations=1)
            for c in label_list
        }

        for ci in label_list:
            mi = dilated[ci]
            for cj in label_list:
                if ci == cj:
                    continue
                if np.logical_and(mi > 0, class_masks[cj] > 0).any():
                    touch[ci, cj] += 1.0

    lam = args.lambda_smooth

    cooc = (n_ij + lam) / (n_i[:, None] + lam * k)
    touch_prob = (touch + lam) / (n_i[:, None] + lam * k)

    prior = args.cooc_weight * cooc + args.touch_weight * touch_prob

    prior[bg, :] = 0.0
    prior[:, bg] = 0.0
    prior[bg, bg] = 1.0

    np.fill_diagonal(prior, np.maximum(np.diag(prior), 1.0))
    prior = row_normalize(prior).astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "prior": torch.from_numpy(prior),
            "cooc": torch.from_numpy(row_normalize(cooc).astype(np.float32)),
            "touch": torch.from_numpy(row_normalize(touch_prob).astype(np.float32)),
            "n_i": torch.from_numpy(n_i.astype(np.float32)),
            "num_classes": k,
            "background_id": bg,
            "ignore_index": ignore,
        },
        out_path,
    )

    print(f"Saved RTB graph prior to: {out_path}")
    print("prior shape:", prior.shape)


if __name__ == "__main__":
    main()
