"""Shared Week 2 helpers, packaged once with Essam's submission.

Extract the complete weekly archive to keep the person folders side by side.
Other scripts load this file from Essam_Week2; do not make per-person copies.
NumPy and Pillow are imported only by the vision functions that need them.
"""

import hashlib
import json
from pathlib import Path
import importlib.util
import random
import sys
from collections import Counter


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def safe_output(person2_root, output):
    """Resolve an output path and prohibit writing into the Person 2 source."""
    root = Path(person2_root).expanduser().resolve()
    target = Path(output).expanduser().resolve()
    if target == root or root in target.parents:
        raise ValueError("Choose an output location outside the Person 2 source folder")
    return target


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


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


PREPROCESSING = "EXIF orientation; alpha on white; RGB; white letterbox 224; ImageNet normalization"


def calibrate_threshold(scores, correct, known):
    """Fit a validation-only boundary between scores, avoiding float equality."""
    import numpy as np

    scores, correct, known = np.asarray(scores), np.asarray(correct, bool), np.asarray(known, bool)
    if not known.any() or known.all():
        return 1.000001, {"status": "insufficient_validation", "balanced_accuracy": None}
    values = sorted(set(scores.tolist()))
    # Midpoints give the same validation decisions without sitting on a score
    # that can shift slightly with the encoder's inference batch size.
    candidates = [-1.000001, 1.000001] + [(left + right) / 2 for left, right in zip(values, values[1:])]
    best = (-1, -1)
    for threshold in candidates:
        accepted = scores >= threshold
        objective = float(((accepted[known] & correct[known]).mean() + (~accepted[~known]).mean()) / 2)
        # Prefer a stricter rejection threshold on ties.
        best = max(best, (objective, threshold))
    return best[1], {"status": "validation_only_pilot", "balanced_accuracy": best[0], "known": int(known.sum()), "unknown": int((~known).sum())}


def classification_report(rows, predictions, labels):
    supported = set(labels)
    true = [row["artifact_id"] if row["artifact_id"] in supported else "__unsupported__" for row in rows]
    confusion = Counter(zip(true, predictions))
    per_class = []
    for label in labels + ["__unsupported__"]:
        support = true.count(label)
        tp = confusion[label, label]
        fp = sum(value for (actual, predicted), value in confusion.items() if actual != label and predicted == label)
        fn = support - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        per_class.append({"artifact_id": label, "support": support, "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / support if support else None, "f1": f1})
    observed = [entry["f1"] or 0.0 for entry in per_class if entry["support"]]
    return {
        "accuracy": sum(a == b for a, b in zip(true, predictions)) / len(rows) if rows else None,
        "macro_f1_observed_classes": sum(observed) / len(observed) if observed else None,
        "sample_count": len(rows), "per_class": per_class,
        "confusion_matrix_sparse": [{"actual": a, "predicted": b, "count": count} for (a, b), count in sorted(confusion.items())],
        "failures": [{"image_id": row["image_id"], "actual": a, "predicted": b} for row, a, b in zip(rows, true, predictions) if a != b],
    }


def load_core(root):
    # Imports must not leave bytecode files in the external source project.
    sys.dont_write_bytecode = True
    path = Path(root) / "museai_collection/core.py"
    spec = importlib.util.spec_from_file_location("person2_vision_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_path(root, row):
    dataset = (Path(root) / "dataset").resolve()
    path = (dataset / "manifests" / row["image"]).resolve()
    if dataset not in path.parents:
        raise ValueError("Image path escapes the dataset")
    return path


def checked_image(root, row, core=None):
    core = core or load_core(root)
    path = image_path(root, row)
    if sha256(path) != row["sha256"]:
        raise ValueError("Image checksum mismatch: " + row["image_id"])
    image = core.decode(path.read_bytes())
    if row.get("pixel_sha256") and core.pixel_hash(image) != row["pixel_sha256"]:
        raise ValueError("Pixel checksum mismatch: " + row["image_id"])
    return image


def augmented_image(image, seed):
    """Small seeded color and rotation changes; no mirrored inscriptions."""
    from PIL import ImageEnhance

    rng = random.Random(seed)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
    return image.rotate(rng.uniform(-8, 8), resample=2, expand=True, fillcolor="white")


def encode_rows(root, rows, encoder, batch_size=8, augmentation_seed=None):
    import numpy as np

    if not rows:
        return np.empty((0, 384), dtype=np.float32)
    core, features = load_core(root), []
    for start in range(0, len(rows), batch_size):
        images = []
        for offset, row in enumerate(rows[start:start + batch_size]):
            image = checked_image(root, row, core)
            if augmentation_seed is not None:
                image = augmented_image(image, augmentation_seed + start + offset)
            images.append(image)
        features.append(encoder.encode(images))
        print(f"Encoded {min(start + batch_size, len(rows))}/{len(rows)} images", flush=True)
    return np.concatenate(features).astype(np.float32)


def export_embeddings(root, output, references, vectors):
    import numpy as np

    np.save(output / "embeddings.npy", vectors, allow_pickle=False)
    write_json(output / "references.json", references)
    write_json(output / "embedding_manifest.json", {
        "schema_version": 1, "source_fingerprint": source_fingerprint(root),
        "encoder": "DINOv2 ViT-S/14", "dimensions": 384, "normalization": "L2",
        "preprocessing": PREPROCESSING, "reference_count": len(references),
        "encoder_weights_sha256": sha256(Path(root) / "assets/dinov2_vits14_pretrain.pth"),
        "encoder_implementation_sha256": sha256(Path(root) / "museai_collection/core.py"),
        "embeddings_sha256": sha256(output / "embeddings.npy"),
        "references_sha256": sha256(output / "references.json"),
        "reference_policy": "Accepted photographs only; scan renders excluded",
    })


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_rows(root, rows):
    mapping = load_mapping(root)
    # Evaluation manifests use null for genuinely unsupported query images.
    return [{**row, "artifact_id": resolve_id(row["artifact_id"], mapping) if row["artifact_id"] is not None else None} for row in rows]


EVALUATION_FILES = {"validation": "validation.json", "pilot_test": "test.json", "tutankhamun_regression": "tutankhamun_test.json"}


ACCEPTED = {"approved", "existing_catalog_reference", "institutional_source_verified"}


def load_inputs(root):
    """Validate the cached reference vectors and preserve the published splits."""
    import numpy as np

    root = Path(root).resolve()
    config = read_json(root / "model/config.json")
    checksums = config.get("checksums", {})
    for name in ("model/embeddings.npy", "model/references.json", "assets/dinov2_vits14_pretrain.pth"):
        if not checksums.get(name) or sha256(root / name) != checksums[name]:
            raise ValueError("Source model checksum mismatch: " + name)
    references = canonical_rows(root, read_json(root / "model/references.json"))
    vectors = np.load(root / "model/embeddings.npy", allow_pickle=False)
    if vectors.shape != (len(references), 384) or not np.isfinite(vectors).all():
        raise ValueError("Invalid source embedding matrix")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
        raise ValueError("Expected unit-normalized DINOv2 embeddings")
    manifest = canonical_rows(root, read_json(root / "dataset/manifests/open_reference.json"))
    expected = {row["image_id"]: row for row in manifest}
    catalog_ids = {row["id"] for row in read_json(root / "dataset/combined_catalog.json")}
    core = load_core(root)
    kept, excluded, indices = [], [], []
    for index, row in enumerate(references):
        source = expected.get(row["image_id"])
        if source is None or any(source.get(key) != row.get(key) for key in ("artifact_id", "sha256", "image", "kind", "capture_group")):
            raise ValueError("Reference vectors do not match the accepted manifest")
        if row["artifact_id"] not in catalog_ids or row.get("split") != "reference" or row.get("review_status") not in ACCEPTED:
            raise ValueError("Reference is not accepted: " + row["image_id"])
        checked_image(root, row, core)
        if row.get("kind") == "scan_render":
            excluded.append({**row, "exclusion_reason": "Synthetic scan renders are excluded from the photographic classifier"})
        else:
            kept.append(row)
            indices.append(index)
    evaluations = {}
    for split, filename in EVALUATION_FILES.items():
        rows = canonical_rows(root, read_json(root / "dataset/manifests" / filename))
        for row in rows:
            if row.get("review_status") not in ACCEPTED:
                raise ValueError("Unaccepted evaluation image")
            checked_image(root, row, core)
        evaluations[split] = rows
    # Fail closed if a newer release introduces duplicate images or capture leakage.
    groups, hashes = {}, {}
    for split, rows in {"train": kept, **evaluations}.items():
        for row in rows:
            for field in ("sha256", "original_sha256", "pixel_sha256"):
                value = row.get(field)
                if value:
                    key = (field, value)
                    if key in hashes and hashes[key] != (split, row["image_id"]):
                        raise ValueError("Duplicate image or split leakage: " + row["image_id"])
                    hashes[key] = (split, row["image_id"])
            group = row.get("capture_group")
            if group:
                if group in groups and groups[group] != split:
                    raise ValueError("Capture group crosses splits: " + group)
                groups[group] = split
    labels = sorted({row["artifact_id"] for row in kept})
    counts = Counter(row["artifact_id"] for row in kept)
    notes = {
        "source_fingerprint": source_fingerprint(root),
        "training_images": len(kept), "supported_classes": len(labels),
        "single_image_classes": sum(value == 1 for value in counts.values()),
        "render_only_classes": sorted({row["artifact_id"] for row in excluded} - set(labels)),
        "metadata_only_classes": sorted(catalog_ids - {row["artifact_id"] for row in references}),
        "split_policy": "All accepted photographic references train the classifier; published validation and pilot manifests stay separate. No random image split within a capture group.",
        "limitation": "Published evaluation images are exposed pilot/regression data. Most classes have one image or one capture group; these results do not estimate generalization to all classes.",
    }
    return kept, vectors[indices].astype(np.float32), evaluations, labels, notes


def make_encoder(root):
    encoder = load_core(root).Encoder(Path(root))
    for parameter in encoder.model.parameters():
        parameter.requires_grad_(False)
    return encoder


def output_directory(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output must be outside the Person 2 source folder and its parent folders")
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_embeddings(root, directory):
    import numpy as np

    directory, root = Path(directory), Path(root)
    metadata = read_json(directory / "embedding_manifest.json")
    if metadata["source_fingerprint"] != source_fingerprint(root):
        raise ValueError("Embedding handoff belongs to a different dataset")
    if metadata.get("encoder") != "DINOv2 ViT-S/14" or metadata.get("preprocessing") != PREPROCESSING:
        raise ValueError("Incompatible embedding encoder or preprocessing")
    for file, key in (("embeddings.npy", "embeddings_sha256"), ("references.json", "references_sha256")):
        if sha256(directory / file) != metadata[key]:
            raise ValueError("Embedding handoff checksum mismatch")
    if sha256(root / "assets/dinov2_vits14_pretrain.pth") != metadata["encoder_weights_sha256"] or sha256(root / "museai_collection/core.py") != metadata["encoder_implementation_sha256"]:
        raise ValueError("Embedding encoder changed since export")
    rows = read_json(directory / "references.json")
    vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
    if vectors.shape != (len(rows), 384) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Invalid handoff embedding matrix")
    if any(row.get("kind") == "scan_render" or row.get("review_status") not in ACCEPTED for row in rows):
        raise ValueError("Handoff includes unaccepted references")
    return rows, vectors, metadata


sha256 = file_sha256
