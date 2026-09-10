"""CodeMixTax: reproducible local-LLM evaluation on the Indic subset.

Five Indian languages (Hindi, Bengali, Marathi, Tamil, Malayalam) across the
eight CodeMixBench tasks -- 18 evaluable splits, 21,256 rows.

Typical use is through the CLI (``cmb-indic run ...``); the library entry points
are re-exported here for notebooks and custom drivers.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import ModelConfig, RunConfig, load_model_config, load_run_config, load_suite
from .registry import (
    DATASETS,
    LANGUAGES,
    TASKS,
    TOTAL_ROWS,
    DatasetSpec,
    coverage_matrix,
    resolve,
)

__all__ = [
    "__version__",
    "DATASETS",
    "LANGUAGES",
    "TASKS",
    "TOTAL_ROWS",
    "DatasetSpec",
    "ModelConfig",
    "RunConfig",
    "coverage_matrix",
    "load_model_config",
    "load_run_config",
    "load_suite",
    "resolve",
]
