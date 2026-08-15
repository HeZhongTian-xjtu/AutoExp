'''CPU-friendly, reproducible House Prices regression template.'''

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / 'configs' / 'experiment.yaml'
DATA_ROOT = Path(os.getenv('AUTOEXP_DATASET_ROOT', str(ROOT / 'data'))).resolve()
PREDICTIONS_PATH = ROOT / 'working' / 'predictions.npz'
RUN_METADATA_PATH = ROOT / 'working' / 'run_metadata.json'


def load_config() -> dict:
    with CONFIG_PATH.open(encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def load_dataset() -> tuple[pd.DataFrame, dict]:
    with (DATA_ROOT / 'dataset_manifest.json').open(encoding='utf-8') as handle:
        dataset_manifest = json.load(handle)
    frame = pd.read_csv(DATA_ROOT / 'train.csv')
    if 'SalePrice' not in frame.columns:
        raise ValueError('the House Prices training asset must contain SalePrice')
    return frame, dataset_manifest


def build_model(config: dict) -> Pipeline:
    frame, _ = load_dataset()
    features = frame.drop(columns=['SalePrice', 'Id'])
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [name for name in features.columns if name not in numeric_columns]
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    categorical = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [('numeric', numeric, numeric_columns), ('categorical', categorical, categorical_columns)],
        remainder='drop',
    )
    estimator = GradientBoostingRegressor(
        learning_rate=float(config['learning_rate']),
        n_estimators=int(config['n_estimators']),
        max_depth=int(config['max_depth']),
        min_samples_leaf=int(config['min_samples_leaf']),
        subsample=float(config['subsample']),
        random_state=int(config['seed']),
        loss='huber',
    )
    return Pipeline([('preprocessor', preprocessor), ('estimator', estimator)])


def main(smoke: bool = False) -> None:
    config = load_config()
    if smoke:
        config = dict(config)
        config['n_estimators'] = min(int(config['n_estimators']), 10)
    frame, dataset_manifest = load_dataset()
    if smoke:
        frame = frame.head(160).copy()
    seed = int(config['seed'])
    features = frame.drop(columns=['SalePrice', 'Id'])
    target = np.log1p(frame['SalePrice'].astype(float).to_numpy())
    train_features, valid_features, train_target, _ = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=seed,
    )
    model = build_model(config)
    started = time.perf_counter()
    model.fit(train_features, train_target)
    predictions = model.predict(valid_features)
    elapsed = time.perf_counter() - started
    parameters = {
        'learning_rate': float(config['learning_rate']),
        'n_estimators': int(config['n_estimators']),
        'max_depth': int(config['max_depth']),
        'min_samples_leaf': int(config['min_samples_leaf']),
        'subsample': float(config['subsample']),
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
        'details': {
            'target': 'SalePrice',
            'split': '80/20 random split by seed',
            'feature_count': int(features.shape[1]),
            'numeric_feature_count': int(features.select_dtypes(include=[np.number]).shape[1]),
            'categorical_feature_count': int(features.select_dtypes(exclude=[np.number]).shape[1]),
            'log_target': True,
        },
    }
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        PREDICTIONS_PATH,
        validation_indices=valid_features.index.to_numpy(dtype=np.int64),
        predictions=np.asarray(predictions, dtype=np.float64),
    )
    with RUN_METADATA_PATH.open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, ensure_ascii=True))


if __name__ == '__main__':
    main('--smoke' in sys.argv[1:])
