"""Shared readers for the Person 2 catalog and safe Week 2 handoffs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_catalog(person2_root):
    """Return canonical records, rejecting blank or repeated artifact IDs."""
    path = Path(person2_root) / "dataset/combined_catalog.json"
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError("The combined catalog must be a list of records")
    ids = [record.get("id") for record in records]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("Every catalog record needs a nonempty string ID")
    if len(ids) != len(set(ids)):
        raise ValueError("The combined catalog contains duplicate artifact IDs")
    return records


def load_mapping(person2_root):
    """Read the explicit legacy-to-canonical mapping without rewriting IDs."""
    path = Path(person2_root) / "dataset/id_mapping.json"
    mapping = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(mapping, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        or not key.strip() or not value.strip()
        for key, value in mapping.items()
    ):
        raise ValueError("The ID mapping must contain nonempty string pairs")
    for key in mapping:
        resolve_id(key, mapping)
    return mapping


def resolve_id(identifier, mapping):
    """Resolve legacy aliases; preserve punctuation and Unicode in stable IDs."""
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Artifact ID must be a nonempty string")
    visited = set()
    while identifier in mapping and mapping[identifier] != identifier:
        if identifier in visited:
            raise ValueError("Cycle in the artifact ID mapping")
        visited.add(identifier)
        identifier = mapping[identifier]
    return identifier


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(person2_root):
    """Bind all handoffs to the same catalog, aliases and ordered RAG chunks."""
    root = Path(person2_root)
    digest = hashlib.sha256()
    for relative in (
        "dataset/combined_catalog.json",
        "dataset/id_mapping.json",
        "dataset/rag/combined_rag_chunks.csv",
    ):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(file_sha256(root / relative).encode("ascii") + b"\n")
    return digest.hexdigest()


def safe_output(person2_root, output):
    """Resolve an output path and prohibit writing into the Person 2 source."""
    root = Path(person2_root).expanduser().resolve()
    target = Path(output).expanduser().resolve()
    if target == root or root in target.parents:
        raise ValueError("Choose an output location outside the Person 2 source folder")
    return target


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
