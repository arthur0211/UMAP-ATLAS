from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any


class ConfigError(ValueError):
    """Invalid or unreadable configuration."""


def _parse_scalar(value: str) -> Any:
    v = value.strip()
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v in {"true", "false"}:
        return v == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.strip().startswith("#"):
            continue
        if not raw_line.startswith(" "):
            key = line.replace(":", "", 1).strip()
            root[key] = {}
            current = root[key]
            continue
        if current is None:
            continue
        key, value = line.strip().split(":", 1)
        current[key.strip()] = _parse_scalar(value)
    return root


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ConfigError("config root must be a mapping")
        return loaded
    except ModuleNotFoundError:
        return _load_simple_yaml(path)


def _required_section(raw: dict[str, Any], section: str) -> dict[str, Any]:
    value = raw.get(section)
    if not isinstance(value, dict):
        raise ConfigError(f"missing required section: {section}")
    return value


def _require(section: dict[str, Any], key: str, typ: type, section_name: str) -> Any:
    if key not in section:
        raise ConfigError(f"missing required key: {section_name}.{key}")
    value = section[key]
    if typ is int:
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"invalid type for {section_name}.{key}: expected int") from None
    if typ is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"invalid type for {section_name}.{key}: expected float") from None
    if typ is str:
        if not isinstance(value, str):
            raise ConfigError(f"invalid type for {section_name}.{key}: expected str")
        return value
    raise ConfigError(f"unsupported type check for {section_name}.{key}")


@dataclass(slots=True)
class RunConfig:
    seed: int
    output_dir: Path
    run_id: str


@dataclass(slots=True)
class DataConfig:
    n_conversations: int
    start_date: str
    days: int
    multi_intent_ratio: float
    incident_spike_ratio: float


@dataclass(slots=True)
class EmbedConfig:
    model: str
    dim: int


@dataclass(slots=True)
class ProjectConfig:
    n_neighbors: int
    min_dist: float
    metric: str


@dataclass(slots=True)
class ClusterConfig:
    min_cluster_size: int
    min_samples: int


@dataclass(slots=True)
class NeighborsConfig:
    k: int


@dataclass(slots=True)
class ExportConfig:
    format: str


@dataclass(slots=True)
class Config:
    run: RunConfig
    data: DataConfig
    embed: EmbedConfig
    project: ProjectConfig
    cluster: ClusterConfig
    neighbors: NeighborsConfig
    export: ExportConfig
    raw: dict[str, Any]

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


def load_config(path: str | Path) -> Config:
    cfg_path = Path(path)
    raw = _load_yaml(cfg_path)

    run = _required_section(raw, "run")
    data = _required_section(raw, "data")
    embed = _required_section(raw, "embed")
    project = _required_section(raw, "project")
    cluster = _required_section(raw, "cluster")
    neighbors = _required_section(raw, "neighbors")
    export = raw.get("export") or {"format": "auto"}
    if not isinstance(export, dict):
        raise ConfigError("invalid type for export section")

    export_format = str(export.get("format", "auto")).lower()
    if export_format not in {"auto", "parquet", "jsonl"}:
        raise ConfigError("export.format must be one of: auto, parquet, jsonl")

    return Config(
        run=RunConfig(
            seed=_require(run, "seed", int, "run"),
            output_dir=Path(_require(run, "output_dir", str, "run")),
            run_id=_require(run, "run_id", str, "run"),
        ),
        data=DataConfig(
            n_conversations=_require(data, "n_conversations", int, "data"),
            start_date=_require(data, "start_date", str, "data"),
            days=_require(data, "days", int, "data"),
            multi_intent_ratio=_require(data, "multi_intent_ratio", float, "data"),
            incident_spike_ratio=_require(data, "incident_spike_ratio", float, "data"),
        ),
        embed=EmbedConfig(
            model=_require(embed, "model", str, "embed"),
            dim=_require(embed, "dim", int, "embed"),
        ),
        project=ProjectConfig(
            n_neighbors=_require(project, "n_neighbors", int, "project"),
            min_dist=_require(project, "min_dist", float, "project"),
            metric=_require(project, "metric", str, "project"),
        ),
        cluster=ClusterConfig(
            min_cluster_size=_require(cluster, "min_cluster_size", int, "cluster"),
            min_samples=_require(cluster, "min_samples", int, "cluster"),
        ),
        neighbors=NeighborsConfig(k=_require(neighbors, "k", int, "neighbors")),
        export=ExportConfig(format=export_format),
        raw=raw,
    )
