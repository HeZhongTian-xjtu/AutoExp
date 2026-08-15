"""Offline Bank Marketing binary classification entrypoint."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv('AUTOEXP_DATASET_ROOT', str(ROOT / 'data'))).resolve()
CONFIG_PATH = ROOT / 'configs' / 'experiment.yaml'
PREDICTIONS_PATH = ROOT / 'working' / 'predictions.npz'
RUN_METADATA_PATH = ROOT / 'working' / 'run_metadata.json'


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8')) or {}


def _load_dataset() -> tuple[pd.DataFrame, dict]:
    dataset_manifest = json.loads((DATA_ROOT / 'dataset_manifest.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(DATA_ROOT / 'train.csv')
    if 'y' not in frame.columns:
        raise ValueError('the Bank Marketing asset must contain y')
    return frame, dataset_manifest


def _build_model(config: dict) -> Pipeline:
    frame, _ = _load_dataset()
    features = frame.drop(columns=['y'])
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [name for name in features.columns if name not in numeric_columns]
    preprocessor = ColumnTransformer(
        [
            ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric_columns),
            ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('one_hot', OneHotEncoder(handle_unknown='ignore'))]), categorical_columns),
        ],
        remainder='drop',
    )
    estimator = LogisticRegression(C=float(config['C']), max_iter=int(config['max_iter']), class_weight='balanced', solver='liblinear', random_state=int(config['seed']))
    return Pipeline([('preprocessor', preprocessor), ('estimator', estimator)])


def main(smoke: bool = False) -> None:
    config = _load_config()
    frame, dataset_manifest = _load_dataset()
    if smoke:
        frame = frame.head(min(3_000, len(frame))).copy()
    seed = int(config['seed'])
    features = frame.drop(columns=['y'])
    target = frame['y'].map({'yes': 1, 'no': 0}).astype(int)
    train_features, valid_features, train_target, _ = train_test_split(features, target, test_size=0.2, random_state=seed, stratify=target)
    model = _build_model(config)
    started = time.perf_counter()
    model.fit(train_features, train_target)
    probabilities = model.predict_proba(valid_features)[:, 1]
    elapsed = time.perf_counter() - started
    parameters = {'C': float(config['C']), 'max_iter': int(config['max_iter']), 'threshold': float(config['threshold'])}
    metadata = {
        'dataset_id': dataset_manifest['dataset_id'],
        'dataset_sha256': dataset_manifest['dataset_sha256'],
        'parameters': parameters,
        'seed': seed,
        'smoke': smoke,
        'training_time_seconds': float(elapsed),
        'train_samples': int(len(train_target)),
        'validation_samples': int(len(valid_features)),
        'details': {'target': 'y', 'positive_label': 'yes', 'split': '80/20 stratified split by seed', 'feature_count': int(features.shape[1])},
    }
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PREDICTIONS_PATH, validation_indices=valid_features.index.to_numpy(dtype=np.int64), probabilities=np.asarray(probabilities, dtype=np.float64))
    RUN_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=True))


if __name__ == '__main__':
    main('--smoke' in sys.argv[1:])
