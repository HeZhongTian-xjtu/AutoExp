'''Trusted evaluator for the House Prices template.

This file is deliberately outside the Agent Patch boundary. Training code may
produce predictions, but it cannot redefine the objective or the metric.
'''

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
METRICS_PATH = ROOT / 'working' / 'metrics.json'


def main() -> None:
    config = _load_yaml(ROOT / 'configs' / 'experiment.yaml')
    metadata = _load_json(ROOT / 'working' / 'run_metadata.json')
    predictions_file = np.load(ROOT / 'working' / 'predictions.npz', allow_pickle=False)
    dataset_manifest = _load_json(DATA_ROOT / 'dataset_manifest.json')
    train_path = DATA_ROOT / 'train.csv'
    actual_dataset_sha256 = _sha256(train_path)
    if actual_dataset_sha256 != dataset_manifest.get('dataset_sha256'):
        raise ValueError('immutable train.csv does not match dataset_manifest.json')

    frame = pd.read_csv(train_path)
    if metadata.get('smoke'):
        frame = frame.head(160).copy()
    indices = predictions_file['validation_indices'].astype(np.int64)
    predictions = predictions_file['predictions'].astype(np.float64)
    if len(indices) != len(predictions) or len(indices) == 0:
        raise ValueError('prediction artifact has an invalid validation split')
    if np.any(indices < 0) or np.any(indices >= len(frame)):
        raise ValueError('prediction indices are outside the registered dataset')

    target = np.log1p(frame.iloc[indices]['SalePrice'].astype(float).to_numpy())
    if len(target) != len(predictions):
        raise ValueError('prediction count does not match evaluator targets')
    parameters = {key: config[key] for key in config if key != 'seed'}
    metrics = {
        'schema_version': '1.0',
        'trial_id': os.getenv('AUTOEXP_TRIAL_ID'),
        'dataset_id': dataset_manifest['dataset_id'],
        'dataset_sha256': actual_dataset_sha256,
        'primary': {
            'name': 'rmse_log',
            'value': float(np.sqrt(mean_squared_error(target, predictions))),
            'direction': 'minimize',
        },
        'secondary': {
            'mae_log': float(mean_absolute_error(target, predictions)),
            'r2_log': float(r2_score(target, predictions)),
            'training_time_seconds': float(metadata.get('training_time_seconds', 0.0)),
            'train_samples': int(metadata.get('train_samples', 0)),
            'validation_samples': int(len(target)),
        },
        'details': {
            **dict(metadata.get('details') or {}),
            'evaluation': 'fixed evaluator over immutable SalePrice targets',
        },
        'parameters': parameters,
        'seed': int(config['seed']),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=True))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()


if __name__ == '__main__':
    main()
