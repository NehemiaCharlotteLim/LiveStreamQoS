"""Inferensi model XGBoost penelitian dari 12 pengukuran QoS browser.

Bundle asli menyimpan pipeline imputer + XGBClassifier dan 24 fitur statistik.
Tidak ada keputusan berbasis ambang QoS yang diterapkan di aplikasi.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / 'models' / 'model_bundle_qos_4parameter_CV_TERBAIK.joblib'
SOURCE_COLUMNS = ('latency_ms', 'jitter_ms', 'packet_loss_percent', 'throughput_mbps')
STATISTICS = ('last', 'mean', 'min', 'max', 'std', 'delta')
FEATURE_NAMES = tuple(f'{stat}_{column}' for stat in STATISTICS for column in SOURCE_COLUMNS)
WINDOW_SIZE = 12
INTERVAL_SECONDS = 5
FORECAST_SECONDS = 60


class ModelSetupError(Exception):
    """Model tidak cocok atau sumber data tidak memenuhi format yang diminta."""


def load_model_bundle(model_path: Path = MODEL_FILE):
    """Memuat bundel tepercaya hasil pelatihan, bukan men-training ulang model.

    File .joblib adalah format pickle. Muat HANYA file Anda sendiri/tepercaya.
    """
    if not model_path.is_file():
        raise ModelSetupError(f'Model XGBoost tidak ditemukan: {model_path.name}.')
    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        raise ModelSetupError(f'Berkas model gagal dimuat: {exc}') from exc
    if not isinstance(bundle, dict):
        raise ModelSetupError('File harus berupa bundle dengan metadata penelitian.')
    model = bundle.get('model')
    if not isinstance(model, Pipeline) or not model.steps or not isinstance(model.steps[-1][1], XGBClassifier):
        raise ModelSetupError('Estimator utama bundle bukan Pipeline dengan XGBClassifier.')
    if bundle.get('best_model_name') != 'XGBoost' or bundle.get('model_name') != 'XGBoost':
        raise ModelSetupError('Metadata tidak menetapkan XGBoost sebagai model terbaik.')
    if tuple(bundle.get('features', [])) != FEATURE_NAMES:
        raise ModelSetupError('Daftar 24 fitur dalam metadata tidak cocok dengan versi aplikasi.')
    if tuple(getattr(model, 'feature_names_in_', [])) != FEATURE_NAMES:
        raise ModelSetupError('Nama/urutan fitur Pipeline tidak cocok.')
    if getattr(model, 'n_features_in_', 0) != 24:
        raise ModelSetupError('Estimator tidak menerima 24 fitur.')
    if (bundle.get('history_intervals') != WINDOW_SIZE or
            bundle.get('interval_seconds') != INTERVAL_SECONDS or
            bundle.get('horizon_intervals') != WINDOW_SIZE):
        raise ModelSetupError('Jendela riwayat dan horizon pelatihan tidak cocok.')
    if bundle.get('feature_version') != 'qos_24_summary_v1':
        raise ModelSetupError('Versi rekayasa fitur belum didukung.')
    if tuple(bundle.get('qos_source_columns', [])) != SOURCE_COLUMNS:
        raise ModelSetupError('Sumber/urutan kolom QoS tidak cocok dengan bundle.')
    if tuple(bundle.get('feature_statistics', [])) != STATISTICS:
        raise ModelSetupError('Daftar statistik fitur tidak cocok dengan bundle.')
    if set(map(int, bundle.get('class_mapping', {}).keys())) != {0, 1}:
        raise ModelSetupError('Model membutuhkan metadata kelas 0 dan 1.')
    if set(map(int, model.classes_)) != {0, 1}:
        raise ModelSetupError('Kelas model berbeda dari metadata bundle.')
    if not math.isclose(float(bundle.get('threshold', 0.5)), 0.5):
        raise ModelSetupError('Ambang klasifikasi model berbeda; periksa protokol training.')
    return model, bundle


def make_feature_frame(samples: Any) -> pd.DataFrame:
    """Dari 12 baris × 4 parameter menjadi 1 baris × 24 fitur.

    Asumsi yang perlu dicocokkan dengan notebook training:
    standar deviasi sampel (ddof=1), delta = nilai terakhir - pertama.
    """
    if not isinstance(samples, list) or len(samples) != WINDOW_SIZE:
        raise ValueError('Input harus tepat 12 sampel, sesuai riwayat model.')
    rows = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f'Sampel ke-{index} harus berupa data parameter QoS.')
        row = {}
        for column in SOURCE_COLUMNS:
            val = sample.get(column)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f'Sampel ke-{index}: {column} tidak valid.')
            row[column] = float(val)
        if row['latency_ms'] < 0 or row['jitter_ms'] < 0 or not (0 <= row['packet_loss_percent'] <= 100) or row['throughput_mbps'] <= 0:
            raise ValueError(f'Sampel ke-{index}: nilai QoS berada di luar rentang valid.')
        rows.append(row)
    df = pd.DataFrame(rows, columns=list(SOURCE_COLUMNS))
    feats = {}
    for statistic in STATISTICS:
        for column in SOURCE_COLUMNS:
            x = df[column].to_numpy(dtype=float)
            if statistic == 'last': result = x[-1]
            elif statistic == 'mean': result = x.mean()
            elif statistic == 'min': result = x.min()
            elif statistic == 'max': result = x.max()
            elif statistic == 'std': result = x.std(ddof=1)
            elif statistic == 'delta': result = x[-1] - x[0]
            feats[f'{statistic}_{column}'] = float(result)
    return pd.DataFrame([feats], columns=list(FEATURE_NAMES))


def predict_window(model, bundle: dict, samples: Any) -> dict[str, Any]:
    """Menghasilkan kelas 0/1 dengan Pipeline.predict(), bukan aturan QoS."""
    x = make_feature_frame(samples)
    try:
        prediction = int(model.predict(x)[0])
        if prediction not in (0, 1):
            raise ModelSetupError(f'Model mengembalikan kelas tidak dikenal: {prediction}.')
        proba = model.predict_proba(x)[0]
        classes = list(map(int, model.classes_))
        prob_pred = float(proba[classes.index(prediction)])
        prob_decrease = float(proba[classes.index(1)])
        if not (math.isfinite(prob_pred) and math.isfinite(prob_decrease)):
            raise ModelSetupError('Probabilitas model tidak valid.')
    except ModelSetupError:
        raise
    except Exception as exc:
        raise ModelSetupError(f'Prediksi XGBoost gagal: {exc}') from exc
    return {
        'label': 'Normal' if prediction == 0 else 'Gangguan',
        'raw_class': prediction,
        'model_label': str(bundle['class_mapping'][prediction]),
        'class_probability': round(prob_pred, 6),
        'probability_qos_decrease': round(prob_decrease, 6),
        'forecast_seconds': FORECAST_SECONDS,
        'history_seconds': WINDOW_SIZE * INTERVAL_SECONDS,
        'model_name': 'XGBoost',
        'feature_count': len(FEATURE_NAMES),
        'feature_values': {k: round(float(x.at[0, k]), 6) for k in FEATURE_NAMES},
    }
