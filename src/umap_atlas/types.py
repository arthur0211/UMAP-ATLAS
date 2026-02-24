from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelineResult:
    rows: list[dict]
    parquet_path: Path
    report_path: Path
