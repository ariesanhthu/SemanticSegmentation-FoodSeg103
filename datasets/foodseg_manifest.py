from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FoodSegManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv,
        data_root,
        train_stage="easy",
        transform=None,
        max_decode_retries: int = 16,
    ):
        """Initialize manifest-driven dataset with robust path resolution.

        Args:
            manifest_csv: CSV file containing training samples.
            data_root: Dataset root directory.
            train_stage: Curriculum stage name.
            transform: Optional image-mask transform callable.
            max_decode_retries: Retry budget when sample decode fails.
        Returns:
            None.
        Raises:
            ValueError: If required manifest columns are missing or stage is invalid.
        """
        self.df = pd.read_csv(manifest_csv)
        self.data_root = Path(data_root)
        self.train_stage = train_stage
        self.transform = transform
        self.max_decode_retries = max(1, int(max_decode_retries))
        self._warned_bad_samples = set()

        required = [
            "stem",
            "split",
            "image_path",
            "mask_path",
            "difficulty_level",
            "stage1_weight",
            "stage2_weight",
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Manifest thiếu cột: {missing}")

        # Chỉ train split
        self.df = self.df[self.df["split"] == "train"].copy()

        # Một CSV, đổi mode là đổi curriculum
        if train_stage == "easy":
            self.df = self.df[
                self.df["difficulty_level"].isin(["easy", "medium"])
            ].copy()
            self.df["sampling_weight"] = self.df["difficulty_level"].map({
                "easy": 4.0,
                "medium": 1.0,
            }).astype(float)

        elif train_stage == "medium":
            self.df["sampling_weight"] = self.df["difficulty_level"].map({
                "easy": 1.5,
                "medium": 3.0,
                "hard": 0.5,
            }).astype(float)

        elif train_stage == "full":
            self.df["sampling_weight"] = 1.0

        elif train_stage == "stage2":
            self.df["sampling_weight"] = self.df["stage2_weight"].astype(float)

        else:
            raise ValueError(f"Unknown train_stage: {train_stage}")

        self.df["sample_id"] = self.df["stem"].astype(str)

        print("=" * 80)
        print(f"FoodSegManifestDataset | stage={train_stage}")
        print(f"samples={len(self.df)}")
        print(self.df["difficulty_level"].value_counts().to_string())
        print("=" * 80)

    def __len__(self):
        """Return number of active samples after stage filtering.

        Args:
            None.
        Returns:
            Number of rows in filtered manifest.
        Raises:
            None.
        """
        return len(self.df)

    def _candidate_paths(self, row, key: str, kind: str) -> list[Path]:
        """Build ordered candidate paths for one image/mask field.

        Args:
            row: Manifest row.
            key: Column name that stores path-like value.
            kind: Either ``"img"`` or ``"mask"``.
        Returns:
            Ordered candidate paths from most to least specific.
        Raises:
            ValueError: If ``kind`` is not supported.
        """
        if kind not in {"img", "mask"}:
            raise ValueError(f"Unsupported kind: {kind}")

        raw_value = str(row[key]).strip()
        path_value = Path(raw_value)
        split = str(row.get("split", "train")).strip() or "train"

        candidates: list[Path] = []

        # 1) Keep absolute path as highest priority.
        if path_value.is_absolute():
            candidates.append(path_value)

        # 2) Interpret as path relative to data_root.
        candidates.append(self.data_root / path_value)

        # 3) If CSV stores train/img/... style, try normalized root join.
        if len(path_value.parts) >= 2:
            candidates.append(self.data_root / Path(*path_value.parts[-3:]))

        # 4) Basename fallback: split/{img|mask}/filename.
        basename = Path(path_value.name)
        candidates.append(self.data_root / split / kind / basename)

        # 5) Cross-split fallback (useful for manifests with wrong split tag).
        other_split = "test" if split == "train" else "train"
        candidates.append(self.data_root / other_split / kind / basename)

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_candidates: list[Path] = []
        for path in candidates:
            key_str = str(path)
            if key_str not in seen:
                unique_candidates.append(path)
                seen.add(key_str)
        return unique_candidates

    def _resolve_existing_path(self, row, key: str, kind: str) -> Path:
        """Resolve an existing filesystem path from manifest fields.

        Args:
            row: Manifest row.
            key: Column name that stores path-like value.
            kind: Either ``"img"`` or ``"mask"``.
        Returns:
            Existing resolved path when found; otherwise best-effort first candidate.
        Raises:
            None.
        """
        candidates = self._candidate_paths(row=row, key=key, kind=kind)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def __getitem__(self, idx):
        """Fetch one sample with decode retry and corruption fallback.

        Args:
            idx: Requested sample index.
        Returns:
            Tuple of ``(image, mask, sample_id)``.
        Raises:
            RuntimeError: If all retries fail to produce a decodable sample.
        """
        last_error = None

        for offset in range(self.max_decode_retries):
            sample_idx = (idx + offset) % len(self.df)
            row = self.df.iloc[sample_idx]

            stem = str(row["sample_id"])
            img_path = self._resolve_existing_path(row=row, key="image_path", kind="img")
            mask_path = self._resolve_existing_path(row=row, key="mask_path", kind="mask")

            try:
                image = np.array(Image.open(img_path).convert("RGB"))
                mask = np.array(Image.open(mask_path), dtype=np.uint8)

                if self.transform is not None:
                    image, mask = self.transform(image, mask)

                return image, mask, stem

            except (
                OSError,
                ValueError,
                SyntaxError,
                RuntimeError,
                UnidentifiedImageError,
            ) as exc:
                last_error = exc
                key = f"{stem}:{img_path}"

                if key not in self._warned_bad_samples:
                    warnings.warn(
                        f"Skipping unreadable sample '{stem}' at '{img_path}': {exc}",
                        RuntimeWarning,
                    )
                    self._warned_bad_samples.add(key)

        raise RuntimeError(
            f"Failed to decode valid sample after {self.max_decode_retries} attempts."
        ) from last_error