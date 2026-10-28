"""Explicit CPU smoke; run directly, outside pytest and its fake-model fixtures.

Use a new dedicated --cache-dir. The online pass is followed by a fresh offline
process. Reports establish operability and structure, never model accuracy.
"""

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_specs import NER_MODEL, NER_REVISION, ZSC_MODEL, ZSC_REVISION  # noqa: E402


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(cache, offline):
    # Set before importing HF/Transformers, even when inherited flags disagree.
    for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME"):
        os.environ[key] = str(cache / key.lower())
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1" if offline else "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if offline:
        def no_network(*args, **kwargs):
            raise RuntimeError("Network attempted during offline smoke")
        socket.socket.connect = no_network
        socket.socket.connect_ex = no_network
        socket.create_connection = no_network
        socket.getaddrinfo = no_network

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )

    torch.set_num_threads(2)
    check(torch.tensor([1.0, 2.0]).numpy().tolist() == [1.0, 2.0], "tensor.numpy")
    report = {
        "offline": offline,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "numpy", "transformers", "tokenizers", "huggingface-hub", "safetensors"
        )},
        "tensor_numpy": "passed",
        "models": [],
    }
    for task, model_id, revision, loader in (
        ("zero-shot-classification", ZSC_MODEL, ZSC_REVISION, AutoModelForSequenceClassification),
        ("token-classification", NER_MODEL, NER_REVISION, AutoModelForTokenClassification),
    ):
        print(f"Loading {model_id}@{revision}, offline={offline}", file=sys.stderr, flush=True)
        common = dict(
            revision=revision, cache_dir=str(cache / "artifacts"),
            local_files_only=offline, trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **common)
        model = loader.from_pretrained(model_id, use_safetensors=True, **common).to("cpu")
        model.eval()
        check(next(model.parameters()).device.type == "cpu", "model must be on CPU")
        kwargs = {"aggregation_strategy": "simple"} if task == "token-classification" else {}
        pipe = pipeline(task, model=model, tokenizer=tokenizer, device=-1, **kwargs)
        entry = {"id": model_id, "revision": revision, "labels": model.config.id2label}
        with torch.inference_mode():
            if task == "zero-shot-classification":
                expected = {"contradiction": 0, "neutral": 1, "entailment": 2}
                check(model.config.label2id == expected, "MNLI mapping mismatch")
                check(pipe.entailment_id == 2, "pipeline entailment mapping mismatch")
                # Pipeline selects contradiction 0 when entailment is the last index.
                check(model.config.id2label[0] == "contradiction", "contradiction mapping")
                outputs = []
                for multi in (False, True):
                    result = pipe(
                        "Oslo kommune behandler en søknad om byggetillatelse.",
                        candidate_labels=["byggesøknad", "sport"], multi_label=multi,
                        hypothesis_template="Dette dokumentet handler om {}.",
                    )
                    check(set(result["labels"]) == {"byggesøknad", "sport"}, "ZSC labels")
                    scores = result["scores"]
                    check(len(scores) == 2 and all(
                        math.isfinite(s) and 0 <= s <= 1 for s in scores
                    ), "ZSC scores")
                    if not multi:
                        check(abs(sum(scores) - 1) < 1e-5, "single-label normalization")
                    outputs.append({"multi_label": multi, "labels": result["labels"],
                                    "scores": scores})
                entry["inference"] = outputs
            else:
                text = "Ola Nordmann arbeider i Oslo kommune i Norge."
                check(tokenizer.is_fast, "NER requires a fast tokenizer for offsets")
                offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
                check(any(b > a for a, b in offsets), "tokenizer offsets missing")
                entities = pipe(text)
                check(bool(entities), "NER smoke needs at least one entity to check structure")
                normalized = []
                for entity in entities:
                    check({"entity_group", "word", "score", "start", "end"} <= entity.keys(),
                          "NER structure")
                    start, end = int(entity["start"]), int(entity["end"])
                    score = float(entity["score"])
                    check(0 <= start < end <= len(text), "NER offsets out of bounds")
                    check(math.isfinite(score) and 0 <= score <= 1, "NER score")
                    normalized.append({
                        "entity_group": entity["entity_group"], "word": entity["word"],
                        "score": score, "start": start, "end": end,
                    })
                # Same JSON boundary as the API's Pydantic response schema.
                from models.ner import NerEntity
                serialized = [NerEntity(**entity).model_dump(mode="json") for entity in normalized]
                json.dumps(serialized, allow_nan=False)
                entry["entities"] = serialized
                entry["offsets"] = "nonempty, bounded offsets on the original smoke input"
        report["models"].append(entry)
        del pipe, model, tokenizer
        gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--offline-only", action="store_true")
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    check(cache.is_dir(), "Create a dedicated cache directory before running the smoke")
    report = run(cache, args.offline_only)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not args.offline_only:
        offline_report = args.report.with_name(args.report.stem + "-offline.json")
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()), "--cache-dir", str(cache),
            "--report", str(offline_report), "--offline-only",
        ], check=True)
    print(f"PASS: {args.report}")


if __name__ == "__main__":
    main()
