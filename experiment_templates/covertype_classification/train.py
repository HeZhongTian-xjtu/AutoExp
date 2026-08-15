"""Offline Covertype multiclass training entrypoint."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv('AUTOEXP_DATASET_ROOT', str(ROOT / 'data'))).resolve()
CONFIG_PATH = ROOT / 'configs' / 'experiment.yaml'
PREDICTIONS_PATH = ROOT / 'working' / 'predictions.npz'
RUN_METADATA_PATH = ROOT / 'working' / 'run_metadata.json'
MAX_FORMAL_ROWS = 120_000
SMOKE_ROWS = 6_000


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8')) or {}


def _load_dataset() -> tuple[pd.DataFrame, dict]:
    dataset_manifest = json.loads((DATA_ROOT / 'dataset_manifest.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(DATA_ROOT / 'train.csv')
    if 'Cover_Type' not in frame.columns:
        raise ValueError('the Covertype asset must contain Cover_Type')
    return frame, dataset_manifest


def main(smoke: bool = False) -> None:
    config = _load_config()
    frame, dataset_manifest = _load_dataset()
    seed = int(config['seed'])
    if smoke:
        frame = frame.head(min(SMOKE_ROWS, len(frame))).copy()
    elif len(frame) > MAX_FORMAL_ROWS:
        frame = frame.sample(n=MAX_FORMAL_ROWS, random_state=seed)
    features = frame.drop(columns=['Cover_Type'])
    target = frame['Cover_Type'].astype(int)
    train_features, valid_features, train_target, _ = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=seed,
        stratify=target,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=float(config['learning_rate']),
        max_iter=int(config['max_iter']) if not smoke else min(int(config['max_iter']), 8),
        max_leaf_nodes=int(config['max_leaf_nodes']),
        min_samples_leaf=int(config['min_samples_leaf']),
        l2_regularization=float(config['l2_regularization']),
        random_state=seed,
    )
    started = time.perf_counter()
    model.fit(train_features, train_target)
    predictions = model.predict(valid_features)
    elapsed = time.perf_counter() - started
    parameters = {
        'learning_rate': float(config['learning_rate']),
        'max_iter': int(config['max_iter']),
        'max_leaf_nodes': int(config['max_leaf_nodes']),
        'min_samples_leaf': int(config['min_samples_leaf']),
        'l2_regularization': float(config['l2_regularization']),
    }
    metadata = {
        'dataset_id': dataset_manifest['dataset_id'],
        'dataset_sha256': dataset_manifest['dataset_sha256'],
        'parameters': parameters,
        'seed': seed,
        'smoke': smoke,
        'training_time_seconds': float(elapsed),
        'train_samples': int(len(train_target)),
        'validation_samples': int(len(valid_features)),
        'selected_rows': int(len(frame)),
        'details': {
            'target': 'Cover_Type',
            'split': '80/20 stratified split by seed',
            'feature_count': int(features.shape[1]),
            'classes': sorted(int(value) for value in target.unique()),
        },
    }
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        PREDICTIONS_PATH,
        validation_indices=valid_features.index.to_numpy(dtype=np.int64),
        predictions=np.asarray(predictions, dtype=np.int64),
    )
    RUN_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=True))


if __name__ == '__main__':
    main('--smoke' in sys.argv[1:])
