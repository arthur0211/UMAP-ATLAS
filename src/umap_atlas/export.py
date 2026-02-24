from __future__ import annotations

from pathlib import Path
import csv
import json


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_parquet_or_csv(rows: list[dict], path: Path) -> str:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
        return "parquet"
    except ModuleNotFoundError:
        fallback = path.with_suffix(".csv")
        with fallback.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                safe_row = {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
                writer.writerow(safe_row)
        return "csv_fallback"


def export_outputs(
    rows: list[dict],
    output_dir: Path,
    run_id: str,
    config_hash: str,
    export_format: str,
    projection_method: str,
    cluster_method: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if export_format == "jsonl":
        data_path = output_dir / f"{run_id}.jsonl"
        _write_jsonl(rows, data_path)
        data_format_used = "jsonl"
    elif export_format == "parquet":
        data_path = output_dir / f"{run_id}.parquet"
        data_format_used = _write_parquet_or_csv(rows, data_path)
        if data_format_used == "csv_fallback":
            data_path = output_dir / f"{run_id}.csv"
    else:  # auto
        data_path = output_dir / f"{run_id}.parquet"
        data_format_used = _write_parquet_or_csv(rows, data_path)
        if data_format_used == "csv_fallback":
            data_path = output_dir / f"{run_id}.csv"

    report_path = output_dir / f"{run_id}_report.md"
    cluster_ids = [int(r["cluster_id"]) for r in rows]
    n_outliers = sum(1 for c in cluster_ids if c == -1)
    n_clusters = len({c for c in cluster_ids if c >= 0})

    lines = [
        f"# Run Report - {run_id}",
        "",
        f"- Rows: {len(rows)}",
        f"- Clusters: {n_clusters}",
        f"- Outliers: {n_outliers}",
        f"- Config hash: {config_hash}",
        f"- Export format used: {data_format_used}",
        f"- Projection method: {projection_method}",
        f"- Cluster method: {cluster_method}",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return data_path, report_path
