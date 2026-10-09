"""
Optional scenario-registry loader.

Reads an optional ``scenarios.yaml`` at the repository root so new scenes can be added
without hand-editing the core dataset code. DatasetM3ED consults it for the list of
sequences to scan and per-sequence frame margins. If the file is absent (the default),
all loaders fall back to the built-in scene configuration — nothing here is required for
the sequences shipped with the project.
"""
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "scenarios.yaml"


def _load() -> dict:
    if yaml is None or not _REGISTRY_PATH.exists():
        return {}
    try:
        with open(_REGISTRY_PATH) as f:
            data = yaml.safe_load(f) or {}
        return data.get("scenarios", {}) or {}
    except Exception:
        return {}


def scenarios() -> dict:
    """All scenarios, keyed by name."""
    return _load()


def get(name: str):
    """Return a single scenario dict (or None)."""
    return _load().get(name)


def registered_sequences() -> list:
    """All train+test sequence names across scenarios (for scene_list)."""
    seqs = []
    for sc in _load().values():
        for key in ("train_seq", "test_seq"):
            s = sc.get(key)
            if s and s not in seqs:
                seqs.append(s)
    return seqs


def margins_for(sequence: str):
    """Return (left, right) frame margins for a sequence, or None if unknown."""
    for sc in _load().values():
        if sequence in (sc.get("train_seq"), sc.get("test_seq")):
            m = sc.get("margins")
            if m:
                return int(m[0]), int(m[1])
    return None


def scenario_for(sequence: str):
    """Return (scenario_name, scenario_dict) that contains this sequence, or (None, None)."""
    for name, sc in _load().items():
        if sequence in (sc.get("train_seq"), sc.get("test_seq")):
            return name, sc
    return None, None
