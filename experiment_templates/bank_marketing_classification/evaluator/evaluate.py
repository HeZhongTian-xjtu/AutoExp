from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
METRICS_PATH = ROOT / 'working' / 'metrics.json'


def main() -> None:
    config = yaml.safe_load((ROOT / 'configs' / 'experiment.yaml').read_text(encoding='utf-8')) or {}
    metadata = json.loads((ROOT / 'working' / 'run_metadata.json').read_text(encoding='utf-8'))
    predictions_file = np.load(ROOT / 'working' / 'predictions.npz', allow_pickle=False)
    dataset_manifest = json.loads((DATA_ROOT / 'dataset_manifest.json').read_text(encoding='utf-8'))
    train_path = DATA_ROOT / 'train.csv'
    actual_sha256 = _sha256(train_path)
    if actual_sha256 != dataset_manifest.get('dataset_sha256'):
        raise ValueError('immutable train.csv does not match dataset_manifest.json')
    frame = pd.read_csv(train_path)
    if metadata.get('smoke'):
        frame = frame.head(min(3_000, len(frame))).copy()
    indices = predictions_file['validation_indices'].astype(np.int64)
    probabilities = predictions_file['probabilities'].astype(np.float64)
    actual = frame.loc[indices, 'y'].map({'yes': 1, 'no': 0}).astype(int).to_numpy()
    if len(actual) != len(probabilities) or len(actual) == 0:
        raise ValueError('prediction artifact has an invalid validation split')
    threshold = float(config['threshold'])
    predicted = (probabilities >= threshold).astype(int)
    metrics = {
        'schema_version': '1.0',
        'trial_id': os.getenv('AUTOEXP_TRIAL_ID'),
        'dataset_id': dataset_manifest['dataset_id'],
        'dataset_sha256': actual_sha256,
        'primary': {'name': 'average_precision', 'value': float(average_precision_score(actual, probabilities)), 'direction': 'maximize'},
        'secondary': {'macro_f1': float(f1_score(actual, predicted, average='macro')), 'accuracy': float(accuracy_score(actual, predicted)), 'threshold': threshold, 'training_time_seconds': float(metadata.get('training_time_seconds', 0.0)), 'validation_samples': int(len(actual))},
        'details': dict(metadata.get('details') or {}),
        'parameters': {key: value for key, value in config.items() if key != 'seed'},
        'seed': int(config['seed']),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()


if __name__ == '__main__':
    main()
