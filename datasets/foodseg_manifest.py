from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FoodSegManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv,
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

        Args:
            None.
        Returns:
            Total number of samples in manifest.
        Raises:
            None.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """Get one sample with robust decode fallback.

        Args:
            idx: Requested sample index.
        Returns:
            Tuple of transformed image, mask, and sample id stem.
        Raises:
            RuntimeError: If no decodable sample is found after retry budget.
        """
        last_error: Exception | None = None

        for offset in range(self.max_decode_retries):
            sample_idx = (idx + offset) % len(self.df)
            row = self.df.iloc[sample_idx]
            stem = str(row["sample_id"])
            img_path = self._path(row, "image_rel_path", "image_path")
            mask_path = self._path(row, "mask_rel_path", "mask_path")

            try:
                image = np.array(Image.open(img_path).convert("RGB"))
                mask = np.array(Image.open(mask_path), dtype=np.uint8)

                if self.transform is not None:
                    image, mask = self.transform(image, mask)

                return image, mask, stem
            except (OSError, ValueError, SyntaxError, RuntimeError, UnidentifiedImageError) as exc:
                last_error = exc
                sample_key = f"{stem}:{img_path}"
                if sample_key not in self._warned_bad_samples:
                    warnings.warn(
                        f"Skipping unreadable sample '{stem}' at '{img_path}': {exc}",
                        RuntimeWarning,
                    )
                    self._warned_bad_samples.add(sample_key)

        raise RuntimeError(
            f"Failed to decode a valid sample after {self.max_decode_retries} attempt(s)."
        ) from last_error
