# Notion task: M1-EM. Owner: Essam.
"""Prepare the current Person 2 data for the Week 1 handoff. Standard library only.

Run: python prepare_data.py --person2-root PERSON2 --output RUN/essam

Writes cleaned catalog records, English knowledge, source provenance, data notes,
field descriptions and a RAG handoff example. These are runtime outputs and are
not included in the source-only submission. Stable IDs are never renamed.
Week 1 preparation preserves supplied facts; it is not independent source review.

Shared helpers: Doha_Week1/week1_common.py. Keep this week's person folders
side by side; extract the complete weekly archive for the runnable layout.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse
import sys


_COMMON_FOLDER = Path(__file__).resolve().parents[1] / "Doha_Week1"
if not (_COMMON_FOLDER / "week1_common.py").is_file():
    raise FileNotFoundError(
        "Missing Doha_Week1/week1_common.py. Extract the complete Week 1 archive, "
        "or extract the person archives into the same folder without renaming their folders."
    )
sys.path.insert(0, str(_COMMON_FOLDER))
from week1_common import fingerprint, load_source, output_path, resolve_id, sha256, write_json


MISSING = {"", "_", "n/a", "none", "null", "unknown", "nan"}
FIELDS = {"name": "Title", "period": "Period", "dynasty": "Dynasty", "date": "Date",
          "material": "Material", "objectNumber": "Museum accession", "civilization": "Civilization",
          "description": "Description"}
DICTIONARY = {
    "id": "Canonical artifact ID in artifact_records.json; identical to artifact_id in the knowledge handoff.",
    "artifact_id": "Canonical object ID; not an image ID or a chunk ID. Preserve punctuation and Unicode.",
    "title": "Supplied English catalog name.", "text": "English catalog facts, with field labels and original values.",
    "source_url": "Supplied museum page URL; its presence does not certify factual verification.",
    "review_status": "Source review status, distinct from data-format checks.",
    "legacy_ids": "Earlier IDs that resolve to this canonical artifact ID.",
    "source_fingerprint": "Checksum binding the current catalog, ID map and source RAG chunks.",
    "has_visual_reference": "Whether the release includes reference images for the object.",
    "license": "Catalog-level attribution, when supplied; image licenses are separate.",
    "category": "Supplied object category or project retrieval label.",
    "source": "Institution or source name supplied with the object.",
    "sourceUrl": "Original catalog spelling of the source_url field used in knowledge records.",
    "source_ids": "Original RAG and CV identifiers retained for provenance; null means no original identifier.",
    "images": "Supplied image URLs and view labels; local verified image paths are prepared by Doha.",
    "visual_reference_count": "Number of reference images declared in the current release.",
    "displayed_in_project": "Source catalog display flag; it does not establish classifier coverage.",
    **{key: label + " as supplied in the catalog; missing values are empty strings." for key, label in FIELDS.items()},
}


def prepare(root, output):
    root = Path(root).resolve()
    output = output_path(root, output)
    catalog, mapping = load_source(root)
    version = fingerprint(root)
    records, knowledge, sources, notes = [], [], [], []
    for original in catalog:
        record = dict(original)
        aid = record["id"]
        for field in [*FIELDS, "category", "source", "sourceUrl", "license"]:
            raw = record.get(field)
            if raw is not None and not isinstance(raw, str):
                raise ValueError(f"{aid}: {field} must be text or null.")
            cleaned = "" if raw is None or raw.strip().casefold() in MISSING else raw.strip()
            record[field] = cleaned
            if cleaned != raw:
                notes.append({"artifact_id": aid, "field": field, "issue": "normalized_missing_or_edge_whitespace", "before": raw, "after": cleaned})
            if not cleaned:
                notes.append({"artifact_id": aid, "field": field, "issue": "missing_value"})
            elif "\ufffd" in cleaned:
                notes.append({"artifact_id": aid, "field": field, "issue": "encoding_replacement_requires_review"})
        parsed = urlparse(record["sourceUrl"])
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"{aid}: a source URL is required for this handoff.")
        text = "\n".join(f"{label}: {record[field]}" for field, label in FIELDS.items() if record[field])
        aliases = sorted(key for key in mapping if key != aid and resolve_id(key, mapping) == aid)
        review = "Supplied catalog text; independent historical verification pending."
        records.append(record)
        knowledge.append({"artifact_id": aid, "title": record["name"], "language": "en", "text": text,
                          "fields": {key: record[key] for key in FIELDS}, "source_url": record["sourceUrl"],
                          "legacy_ids": aliases, "review_status": review})
        sources.append({"artifact_id": aid, "source": record["source"], "source_url": record["sourceUrl"],
                        "license": record["license"], "source_ids": record.get("source_ids", {}),
                        "legacy_ids": aliases, "catalog_file": "dataset/combined_catalog.json", "review_status": review})
    with (root / "dataset/rag/combined_rag_chunks.csv").open(encoding="utf-8-sig", newline="") as stream:
        chunks = list(csv.DictReader(stream))
    ids = {row["id"] for row in records}
    chunk_ids = [row["Chunk ID"] for row in chunks]
    if len(set(chunk_ids)) != len(chunk_ids) or {row["Artifact ID"] for row in chunks} != ids:
        raise ValueError("Source RAG chunks must have unique chunk IDs and cover this catalog exactly.")
    if any(not row["Chunk Text"].strip() for row in chunks):
        raise ValueError("The source contains blank RAG chunks.")
    lineage = {"source_fingerprint": version, "catalog_sha256": sha256(root / "dataset/combined_catalog.json"),
               "source_chunks_sha256": sha256(root / "dataset/rag/combined_rag_chunks.csv"),
               "source_chunk_count": len(chunks), "artifact_count": len(records),
               "review_note": "Source links and format checks do not establish historical approval."}
    write_json(output / "artifact_records.json", {**lineage, "records": records})
    write_json(output / "knowledge_en.json", {**lineage, "records": knowledge})
    write_json(output / "sources.json", {**lineage, "records": sources})
    write_json(output / "data_notes.json", {**lineage, "issues": notes, "issue_counts": dict(Counter(row["issue"] for row in notes))})
    write_json(output / "data_dictionary.json", DICTIONARY)
    example = next((row for row in knowledge if row["fields"].get("material")), knowledge[0] if knowledge else None)
    write_json(output / "handoff_example.json", {**lineage, "recipient": "Adel", "knowledge_record": example,
               "question": "What material is this artifact made of?", "input_file": "knowledge_en.json",
               "rebuild_policy": "Rebuild the Week 1 index whenever this knowledge file or source fingerprint changes."})
    if fingerprint(root) != version:
        raise RuntimeError("Source changed during preparation; rerun before handing off.")
    return {"artifacts": len(records), "source_chunks": len(chunks),
            "missing_descriptions": sum(not row["description"] for row in records), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.person2_root, args.output))


if __name__ == "__main__":
    main()
