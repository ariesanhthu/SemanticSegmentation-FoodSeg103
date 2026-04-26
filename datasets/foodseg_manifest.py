from pathlib import Path
import warnings
<<<<<<< HEAD

import numpy as np
import pandas as pd
=======
import pandas as pd
import numpy as np
>>>>>>> 4b2263ec99bfd68c12e9df2ee5b7100f626d0ed2
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FoodSegManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv,
<<<<<<< HEAD
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
=======
        project_root=".",
        transform=None,
        max_decode_retries: int = 16,
    ):
        """Initialize dataset from manifest CSV.

        Args:
            manifest_csv: Path to manifest CSV file.
            project_root: Root directory used for relative paths.
            transform: Transform callable applied to image and mask.
            max_decode_retries: Number of fallback samples to try when decoding fails.
        Returns:
            None.
        Raises:
            ValueError: If max_decode_retries is less than 1.
        """
        self.df = pd.read_csv(manifest_csv)
        self.project_root = Path(project_root)
        self.transform = transform
        self.max_decode_retries = max(1, int(max_decode_retries))
        self._warned_bad_samples: set[str] = set()

        if "sample_id" not in self.df.columns:
            self.df["sample_id"] = [f"sample_{i:08d}" for i in range(len(self.df))]

        if "sampling_weight" not in self.df.columns:
            self.df["sampling_weight"] = 1.0

    def _path(self, row, rel_col, abs_col):
        """Resolve file path from relative or absolute manifest columns.

        Args:
            row: Manifest row.
            rel_col: Relative path column name.
            abs_col: Absolute path column name.
        Returns:
            Path object pointing to resolved file.
        Raises:
            KeyError: If required path columns are missing.
        """
        if rel_col in self.df.columns and isinstance(row[rel_col], str) and row[rel_col]:
            return self.project_root / row[rel_col]
        return Path(row[abs_col])

    def __len__(self):
        """Return dataset size.
>>>>>>> 4b2263ec99bfd68c12e9df2ee5b7100f626d0ed2

        Args:
            None.
        Returns:
<<<<<<< HEAD
            Number of rows in filtered manifest.
=======
            Total number of samples in manifest.
>>>>>>> 4b2263ec99bfd68c12e9df2ee5b7100f626d0ed2
        Raises:
            None.
        """
        return len(self.df)

<<<<<<< HEAD
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
=======
    def __getitem__(self, idx):
        """Get one sample with robust decode fallback.
>>>>>>> 4b2263ec99bfd68c12e9df2ee5b7100f626d0ed2

        Args:
            idx: Requested sample index.
        Returns:
<<<<<<< HEAD
            Tuple of ``(image, mask, sample_id)``.
        Raises:
            RuntimeError: If all retries fail to produce a decodable sample.
        """
        last_error = None
=======
            Tuple of transformed image, mask, and sample id stem.
        Raises:
            RuntimeError: If no decodable sample is found after retry budget.
        """
        last_error: Exception | None = None
>>>>>>> 4b2263ec99bfd68c12e9df2ee5b7100f626d0ed2

        for offset in range(self.max_decode_retries):
            sample_idx = (idx + offset) % len(self.df)
            row = self.df.iloc[sample_idx]
<<<<<<< HEAD

            stem = str(row["sample_id"])
            img_path = self._resolve_existing_path(row=row, key="image_path", kind="img")
            mask_path = self._resolve_existing_path(row=row, key="mask_path", kind="mask")
=======
            stem = str(row["sample_id"])
            img_path = self._path(row, "image_rel_path", "image_path")
            mask_path = self._path(row, "mask_rel_path", "mask_path")
>>>>>>> 4b2263ec99bfd68c12e9df2ee5b7100f626d0ed2

            try:
                image = np.array(Image.open(img_path).convert("RGB"))
                mask = np.array(Image.open(mask_path), dtype=np.uint8)

                if self.transform is not None:
                    image, mask = self.transform(image, mask)

                return image, mask, stem
<<<<<<< HEAD

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
=======
            except (OSError, ValueError, SyntaxError, RuntimeError, UnidentifiedImageError) as exc:
                last_error = exc
                sample_key = f"{stem}:{img_path}"
                if sample_key not in self._warned_bad_samples:
>>>>>>> 4b2263ec99bfd68c12e9df2ee5b7100f626d0ed2
                    warnings.warn(
                        f"Skipping unreadable sample '{stem}' at '{img_path}': {exc}",
                        RuntimeWarning,
                    )
<<<<<<< HEAD
                    self._warned_bad_samples.add(key)

        raise RuntimeError(
            f"Failed to decode valid sample after {self.max_decode_retries} attempts."
        ) from last_error
=======
                    self._warned_bad_samples.add(sample_key)

        raise RuntimeError(
            f"Failed to decode a valid sample after {self.max_decode_retries} attempt(s)."
        ) from last_error
>>>>>>> 4b2263ec99bfd68c12e9df2ee5b7100f626d0ed2
