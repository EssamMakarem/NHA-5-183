"""Validate the handoff corpus and preserve source spans while chunking."""
import hashlib
import json
import re
from pathlib import Path


def load_records(path):
    records = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    seen = set()
    for record in records:
        for field in ("artifact_id", "title", "language", "text", "source_url"):
            if not isinstance(record.get(field), str):
                raise ValueError(f"Missing string field: {field}")
        if record["artifact_id"] in seen:
            raise ValueError("Duplicate artifact ID")
        seen.add(record["artifact_id"])
        if record["language"] != "en":
            raise ValueError("The Week 1 baseline accepts English records only")
        if not record["source_url"].startswith(("https://", "http://")):
            raise ValueError("Every record requires a source URL")
    return records


def chunk_records(records, max_words=80, overlap_words=16):
    if max_words <= overlap_words or overlap_words < 0:
        raise ValueError("Require 0 <= overlap_words < max_words")
    chunks = []
    for record in records:
        text = record["text"]
        if text.strip().lower() in {"", "_", "n/a", "unknown", "none"}:
            continue
        words = list(re.finditer(r"\S+", text))
        start = 0
        number = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            left, right = words[start].start(), words[end - 1].end()
            chunks.append({
                "chunk_id": f"{record['artifact_id']}:{number:03d}",
                "artifact_id": record["artifact_id"],
                "title": record["title"],
                "text": text[left:right],
                "start_char": left,
                "end_char": right,
                "source_url": record["source_url"],
            })
            if end == len(words):
                break
            start = end - overlap_words
            number += 1
    return chunks


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
