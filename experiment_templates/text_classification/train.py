"""Small deterministic text classification template for the AutoExp local loop."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
METRICS_PATH = ROOT / "working" / "metrics.json"
DATA_ROOT = Path(os.getenv('AUTOEXP_DATASET_ROOT', str(ROOT / 'data'))).resolve()


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset() -> tuple[list[str], np.ndarray, dict]:
    texts = [
        "the new release improves model accuracy and validation results",
        "training completed with a stable loss curve and better recall",
        "the experiment reports a strong improvement over the baseline",
        "the classifier performs well on clean validation examples",
        "the optimizer converges quickly after the parameter update",
        "the research result supports the proposed hypothesis",
        "the measured score is higher and the error is lower",
        "the model generalizes well to unseen positive examples",
        "the data pipeline produces reliable training features",
        "the evaluation confirms a useful improvement in performance",
        "the baseline achieves a reasonable score on the task",
        "the final report contains reproducible experiment evidence",
        "the prediction is noisy and the validation error is high",
        "the training run fails to improve the classification score",
        "the model overfits the small sample and misses new cases",
        "the experiment shows weak recall on difficult examples",
        "the optimization becomes unstable with an aggressive setting",
        "the result does not support the original hypothesis",
        "the measured score drops and the error becomes larger",
        "the classifier performs poorly on unseen negative examples",
        "the data pipeline contains noisy and unreliable features",
        "the evaluation reveals a regression from the baseline",
        "the run stops early because the validation result is weak",
        "the report describes a failed experiment and its cause",
    ]
    labels = np.array([1] * 12 + [0] * 12)
    return texts, labels, {}


def load_dataset() -> tuple[list[str], np.ndarray, dict]:
    train_path = DATA_ROOT / 'train.csv'
    if not train_path.is_file():
        return build_dataset()
    frame = pd.read_csv(train_path)
    required = {'text', 'label'}
    if not required.issubset(frame.columns):
        raise ValueError('custom text dataset train.csv must contain text and label columns')
    metadata_path = DATA_ROOT / 'dataset_manifest.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
    return frame['text'].fillna('').astype(str).tolist(), frame['label'].tolist(), metadata


def main(smoke: bool = False) -> None:
    config = load_config()
    if smoke:
        config = dict(config)
        config['epochs'] = 1
    seed = int(config["seed"])
    texts, labels, dataset_metadata = load_dataset()
    if smoke:
        texts = texts[:8] + texts[-8:]
        labels = np.array([1] * 8 + [0] * 8)
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=seed, stratify=labels
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    train_features = vectorizer.fit_transform(train_texts)
    valid_features = vectorizer.transform(valid_texts)

    start = time.perf_counter()
    classifier = LogisticRegression(
        C=max(
            0.01,
            min(
                100.0,
                float(config["learning_rate"]) * 1000
                / (1 + float(config["dropout"]) * int(config["lora_rank"])),
            ),
        ),
        max_iter=max(1, int(config["epochs"]) * int(config["batch_size"])),
        tol=1e-4,
        random_state=seed,
        solver='liblinear',
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(valid_features)
    elapsed = time.perf_counter() - start
    primary_value = float(f1_score(valid_labels, predictions, average="macro"))

    parameters = {
        "learning_rate": float(config["learning_rate"]),
        "batch_size": int(config["batch_size"]),
        "dropout": float(config["dropout"]),
        "lora_rank": int(config["lora_rank"]),
        "epochs": int(config["epochs"]),
    }
    metrics = {
        "schema_version": "1.0",
        "primary": {"name": "macro_f1", "value": primary_value, "direction": "maximize"},
        "secondary": {"training_time_seconds": elapsed, "validation_samples": len(valid_labels)},
        "parameters": parameters,
        "seed": seed,
    }
    if dataset_metadata:
        metrics['dataset_id'] = dataset_metadata.get('dataset_id')
        metrics['dataset_sha256'] = dataset_metadata.get('dataset_sha256')
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, ensure_ascii=True))


if __name__ == "__main__":
    main('--smoke' in sys.argv[1:])
