"""Train or run a frozen DINOv2 encoder with a trainable artifact classifier.

Install numpy, Pillow and torch. Use the complete Person 2 v5.4 project as input.
Example: python train_classifier.py train --person2-root PATH --output RUN
Then: python train_classifier.py predict --person2-root PATH --model-dir RUN --image PHOTO
The output directory holds generated checkpoints, embeddings and pilot reports.
Run fixed epochs; do not select a checkpoint using the exposed test manifests.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vision_data import (PREPROCESSING, calibrate_threshold, classification_report,
    encode_rows, export_embeddings, load_core, load_inputs, make_encoder,
    output_directory, read_json, sha256, source_fingerprint, write_json)


def train(args):
    root = Path(args.person2_root).resolve()
    output = output_directory(root, args.output)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    print("Checking accepted images, source checksums and published splits...", flush=True)
    rows, vectors, evaluations, labels, quality = load_inputs(root)
    print(f"Accepted {len(rows)} photographs across {len(labels)} artifact IDs", flush=True)
    encoder = make_encoder(root)
    spot_indices = sorted({0, len(rows) // 2, len(rows) - 1})
    recomputed = encode_rows(root, [rows[index] for index in spot_indices], encoder, args.batch_size)
    cached = vectors[spot_indices]
    if not np.allclose(recomputed, cached, atol=1e-4, rtol=1e-4):
        raise ValueError("Cached vectors disagree with the current encoder and preprocessing")
    quality["encoder_spotcheck"] = {"image_ids": [rows[index]["image_id"] for index in spot_indices], "maximum_absolute_difference": float(np.abs(recomputed - cached).max()), "minimum_cosine": float((recomputed * cached).sum(axis=1).min())}
    export_embeddings(root, output, rows, vectors)
    write_json(output / "split_manifest.json", {"source_fingerprint": quality["source_fingerprint"], "seed": args.seed, "train": rows, **evaluations})
    write_json(output / "data_quality.json", quality)
    lookup = {label: index for index, label in enumerate(labels)}
    targets = np.array([lookup[row["artifact_id"]] for row in rows])
    features, all_targets = [vectors], [targets]
    for view in range(args.augmentation_views):
        features.append(encode_rows(root, rows, encoder, args.batch_size, args.seed + view * len(rows)))
        all_targets.append(targets)
    x = torch.from_numpy(np.concatenate(features))
    y = torch.from_numpy(np.concatenate(all_targets)).long()
    # A class prototype starts the linear head; AdamW then learns its parameters.
    head = torch.nn.Linear(384, len(labels))
    with torch.no_grad():
        means = torch.stack([x[y == index].mean(dim=0) for index in range(len(labels))])
        head.weight.copy_(torch.nn.functional.normalize(means, dim=1) * 10)
        head.bias.zero_()
    frequencies = torch.bincount(y, minlength=len(labels)).float()
    weights = (len(y) / (len(labels) * frequencies)).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    history = []
    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(x), y)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(loss.detach())})
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={float(loss.detach()):.5f}", flush=True)
    head.eval()
    predictions = {}
    for split, split_rows in evaluations.items():
        features = encode_rows(root, split_rows, encoder, args.batch_size)
        with torch.inference_mode():
            probabilities = head(torch.from_numpy(features)).softmax(dim=1).numpy()
        indices = probabilities.argmax(axis=1)
        predictions[split] = {"ids": [labels[index] for index in indices], "scores": probabilities.max(axis=1).astype(float).tolist()}
    val = predictions["validation"]
    known = [row["artifact_id"] in lookup for row in evaluations["validation"]]
    correct = [row["artifact_id"] == prediction for row, prediction in zip(evaluations["validation"], val["ids"])]
    threshold, calibration = calibrate_threshold(val["scores"], correct, known)
    reports = {}
    for split, split_rows in evaluations.items():
        result = predictions[split]
        accepted = [aid if score >= threshold else "__unsupported__" for aid, score in zip(result["ids"], result["scores"])]
        reports[split] = {
            "open_set": classification_report(split_rows, accepted, labels),
            "known_closed_set": classification_report([row for row in split_rows if row["artifact_id"] in lookup], [aid for row, aid in zip(split_rows, result["ids"]) if row["artifact_id"] in lookup], labels),
            "predictions": [{"image_id": row["image_id"], "expected_id": row["artifact_id"], "candidate_id": aid, "softmax_score": score, "accepted": score >= threshold} for row, aid, score in zip(split_rows, result["ids"], result["scores"])],
        }
    torch.save({"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu()}, output / "classifier.pt")
    write_json(output / "labels.json", labels)
    write_json(output / "inference_config.json", {
        "schema_version": 1, "source_fingerprint": quality["source_fingerprint"],
        "encoder": "DINOv2 ViT-S/14 frozen", "classifier": "trained linear head",
        "preprocessing": PREPROCESSING, "dimensions": 384, "reject_below": threshold,
        "threshold_calibration": calibration, "seed": args.seed, "epochs": args.epochs,
        "threshold_policy": "Validation score midpoints; balanced known-correct acceptance and unknown rejection; stricter boundary wins ties",
        "learning_rate": args.learning_rate, "augmentation_views": args.augmentation_views,
        "augmentation": "Brightness/contrast 0.9-1.1 and rotation -8 to 8 degrees, seeded; no horizontal flip",
        "encoder_weights_sha256": sha256(root / "assets/dinov2_vits14_pretrain.pth"),
        "encoder_implementation_sha256": sha256(root / "museai_collection/core.py"),
        "checkpoint_sha256": sha256(output / "classifier.pt"), "labels_sha256": sha256(output / "labels.json"),
        "confidence_note": "Softmax score is not a calibrated probability of identity. Accepted outputs are possible matches.",
    })
    with torch.inference_mode():
        train_predictions = [labels[index] for index in head(torch.from_numpy(vectors)).argmax(dim=1).tolist()]
    write_json(output / "evaluation.json", {"source_fingerprint": quality["source_fingerprint"], "scope": quality["limitation"], "threshold_calibration": calibration, "training_history": history, "training_diagnostic": classification_report(rows, train_predictions, labels), **reports})
    print(json.dumps({"output": str(output), "training_images": len(rows), "classes": len(labels), "threshold": threshold, "pilot_closed_set_accuracy": reports["pilot_test"]["known_closed_set"]["accuracy"]}), flush=True)


def predict(args):
    root, directory = Path(args.person2_root).resolve(), Path(args.model_dir)
    config = read_json(directory / "inference_config.json")
    if config["source_fingerprint"] != source_fingerprint(root):
        raise ValueError("Classifier belongs to a different dataset")
    for path, key in ((directory / "classifier.pt", "checkpoint_sha256"), (directory / "labels.json", "labels_sha256"), (root / "assets/dinov2_vits14_pretrain.pth", "encoder_weights_sha256"), (root / "museai_collection/core.py", "encoder_implementation_sha256")):
        if sha256(path) != config[key]:
            raise ValueError("Classifier or encoder checksum mismatch")
    labels = read_json(directory / "labels.json")
    head = torch.nn.Linear(384, len(labels))
    head.load_state_dict(torch.load(directory / "classifier.pt", map_location="cpu", weights_only=True))
    head.eval()
    vector = make_encoder(root).encode([load_core(root).decode(Path(args.image).read_bytes())])
    with torch.inference_mode():
        scores = head(torch.from_numpy(vector)).softmax(dim=1)[0]
    score, index = scores.max(dim=0)
    accepted = float(score) >= config["reject_below"]
    print(json.dumps({"status": "possible_match" if accepted else "insufficient_visual_evidence", "artifact_id": labels[int(index)] if accepted else None, "candidate_id": labels[int(index)], "softmax_score": float(score), "reject_below": config["reject_below"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train the head and export embeddings plus pilot evaluation")
    training.add_argument("--person2-root", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--learning-rate", type=float, default=0.03)
    training.add_argument("--augmentation-views", type=int, default=1, help="Seeded augmented views per photograph; 0 is a quick baseline")
    training.add_argument("--batch-size", type=int, default=8)
    inference = commands.add_parser("predict", help="Predict with a saved classifier")
    inference.add_argument("--person2-root", required=True)
    inference.add_argument("--model-dir", required=True)
    inference.add_argument("--image", required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1 or args.augmentation_views < 0 or args.batch_size < 1 or args.learning_rate <= 0:
            parser.error("epochs, batch-size and learning-rate must be positive; augmentation-views cannot be negative")
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
