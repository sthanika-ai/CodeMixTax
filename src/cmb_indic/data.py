"""Dataset loading with a local, content-addressed cache.

Splits are fetched once from the Hugging Face Hub into ``data/raw/<task>/<name>.csv``
and read from there afterwards, so a full sweep does no network I/O and an
offline machine can reproduce a run. The Hub revision actually used is recorded
in ``data/raw/_manifest.json`` and copied into every run's metadata, which is
what makes a published result pinnable to an exact dataset state.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .registry import HF_DATASET_ID, DatasetSpec

MANIFEST_NAME = "_manifest.json"


@dataclass
class Split:
    """A loaded split, normalised into the columns the pipeline needs."""

    spec: DatasetSpec
    frame: pd.DataFrame
    #: Row index values (from the CSV's own `index` column) in frame order.
    indices: list[int]
    #: Whether `frame` is a subsample of the full split.
    subsampled: bool
    n_full: int
    sha256: str

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def gold(self) -> list:
        """Gold answers: label sequences for token tasks, else scalar values."""
        if self.spec.metric == "token_label":
            return [parse_token_field(v) for v in self.frame["answer"]]
        return list(self.frame["answer"])

    @property
    def tokens(self) -> list[list[str]] | None:
        if "tokens" not in self.frame.columns:
            return None
        return [parse_token_field(v) for v in self.frame["tokens"]]


def parse_token_field(value) -> list[str]:
    """Parse a stringified Python list from the CSV into a list of strings.

    The `tokens` and token-task `answer` columns are stored as the repr of a
    Python list, e.g. ``"['Ladke', 'Ne', 'ek']"``.
    """
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        # Last resort: treat it as whitespace-separated.
        return text.split()
    if isinstance(parsed, (list, tuple)):
        return [str(x) for x in parsed]
    return [str(parsed)]


# --------------------------------------------------------------------------- #
# Download / cache
# --------------------------------------------------------------------------- #
def local_path(spec: DatasetSpec, data_dir: Path) -> Path:
    return Path(data_dir) / spec.hf_path


def ensure_downloaded(
    spec: DatasetSpec,
    data_dir: Path,
    *,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Make sure the split's CSV exists locally; download it if not.

    Uses ``huggingface_hub.hf_hub_download`` rather than ``datasets.load_dataset``
    because we want the raw CSV bytes on disk (checksummable, greppable, and
    independent of the `datasets` cache layout).
    """
    dest = local_path(spec, data_dir)
    if dest.exists() and not force:
        return dest

    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    fetched = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=spec.hf_path,
        repo_type="dataset",
        revision=revision,
        token=os.environ.get("HF_TOKEN") or None,
    )
    dest.write_bytes(Path(fetched).read_bytes())
    return dest


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(data_dir: Path, entries: dict[str, dict]) -> Path:
    """Record what was downloaded, with checksums, for reproducibility."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / MANIFEST_NAME
    existing: dict = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.setdefault("dataset", HF_DATASET_ID)
    existing.setdefault("splits", {})
    existing["splits"].update(entries)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_manifest(data_dir: Path) -> dict:
    path = Path(data_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_split(
    spec: DatasetSpec,
    data_dir: Path,
    *,
    limit: int | None = None,
    seed: int = 1234,
    sample: str = "head",
    revision: str | None = None,
    download: bool = True,
) -> Split:
    """Load one split, optionally subsampled.

    Args:
        limit: keep at most this many rows. `None` means the full split.
        sample: ``"head"`` takes the first `limit` rows (fast, deterministic,
            good for smoke tests); ``"random"`` takes a seeded random sample,
            which is what you want for an honest partial evaluation because
            several splits are ordered by label.
        revision: pin a Hub revision (commit sha or tag).

    Raises:
        FileNotFoundError: if the CSV is absent and `download` is False.
    """
    path = local_path(spec, data_dir)
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} is missing. Run `cmb-indic download` first, "
                "or pass download=True."
            )
        path = ensure_downloaded(spec, data_dir, revision=revision)

    frame = pd.read_csv(path, encoding="utf-8")
    n_full = len(frame)

    if n_full != spec.n_rows:
        # Not fatal: the dataset may legitimately have been updated upstream.
        # Loud, because it invalidates comparisons against recorded numbers.
        import warnings

        warnings.warn(
            f"{spec.name}: loaded {n_full} rows but the registry expects "
            f"{spec.n_rows}. The Hub copy may have changed; re-check row counts "
            f"before publishing (`cmb-indic validate --check-rows`).",
            stacklevel=2,
        )

    _require_columns(spec, frame)

    subsampled = False
    if limit is not None and limit < n_full:
        subsampled = True
        if sample == "random":
            frame = frame.sample(n=limit, random_state=seed).sort_values("index")
        elif sample == "head":
            frame = frame.head(limit)
        else:
            raise ValueError(f"sample must be 'head' or 'random', got {sample!r}")

    frame = frame.reset_index(drop=True)
    return Split(
        spec=spec,
        frame=frame,
        indices=[int(i) for i in frame["index"]],
        subsampled=subsampled,
        n_full=n_full,
        sha256=sha256_of(path),
    )


def _require_columns(spec: DatasetSpec, frame: pd.DataFrame) -> None:
    needed = {"index", "answer", spec.input_col}
    if spec.task == "gsm8k":
        needed.add("cot")  # needed to build few-shot CoT exemplars
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(
            f"{spec.name}: CSV is missing required column(s) {sorted(missing)}; "
            f"found {list(frame.columns)}"
        )
