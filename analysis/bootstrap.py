"""Analyze aligned predictions with paired within-WSI patch bootstrap sampling."""
import argparse
import json
from pathlib import Path

def analyze(prediction_dir, output_dir, models, replicates=1000, seed=2026):
    import numpy as np
    import pandas as pd
    from .statistics import (PROBABILITY_COLUMNS, MODEL_ORDER, bootstrap_indices, dominant_reference,
                             bootstrap_model, summarize, paired_differences, point_wsi_metrics)
    if replicates < 1 or not models or len(models) != len(set(models)) or not set(models).issubset(MODEL_ORDER):
        raise ValueError('Invalid replicates or model selection.')
    manifest = None
    probabilities = {}
    for name in models:
        frame = pd.read_csv(Path(prediction_dir) / f'{name}_tile_predictions.csv', dtype={'sample_id': str, 'patient_id': str, 'wsi_id': str})
        required = ['sample_id', 'patient_id', 'wsi_id', 'label'] + PROBABILITY_COLUMNS
        if not set(required).issubset(frame.columns) or frame.empty or frame[required].isna().any().any():
            raise ValueError('Predictions require nonempty tile, patient, WSI, label and six probability columns.')
        if frame['sample_id'].duplicated().any():
            raise ValueError('Duplicate sample IDs.')
        if not frame['label'].isin(range(6)).all():
            raise ValueError('Labels must be integers 0..5.')
        if manifest is None:
            manifest = frame[['sample_id', 'patient_id', 'wsi_id', 'label']].copy().reset_index(drop=True)
        else:
            if set(frame['sample_id']) != set(manifest['sample_id']):
                raise ValueError('Every model must evaluate exactly the same tiles.')
            frame = frame.set_index('sample_id').loc[manifest['sample_id']].reset_index()
            if not frame[['patient_id', 'wsi_id', 'label']].equals(manifest[['patient_id', 'wsi_id', 'label']]):
                raise ValueError('Patient, WSI or label differs between model predictions.')
        probs = frame[PROBABILITY_COLUMNS].to_numpy(np.float32)
        if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-5):
            raise ValueError('Invalid class probabilities.')
        probabilities[name] = probs
    if manifest.groupby('wsi_id')['patient_id'].nunique().max() != 1:
        raise ValueError('A WSI cannot belong to multiple patients.')
    if manifest.groupby('patient_id')['wsi_id'].nunique().max() != 1:
        raise ValueError('This study protocol requires one WSI per patient; no multi-slide aggregation is implemented.')
    reference = dominant_reference(manifest)
    if (~reference['is_tie']).sum() == 0:
        raise ValueError('All WSI reference labels are tied; no evaluable WSIs.')
    samples = bootstrap_indices(manifest, replicates, seed)
    frames, points = [], []
    for name, probs in probabilities.items():
        _, wsi, _ = bootstrap_model(name, probs, manifest, samples, reference, replicates)
        frames.append(wsi)
        points.append({'model': name, **point_wsi_metrics(probs, manifest, reference)})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(output_dir / 'bootstrap_wsi_metrics.csv', index=False)
    summarize(metrics, 'wsi').to_csv(output_dir / 'bootstrap_wsi_summary.csv', index=False)
    pd.DataFrame(points).to_csv(output_dir / 'wsi_point_estimates.csv', index=False)
    if 'mixscon_fl' in models and len(models) > 1:
        paired_differences(metrics).to_csv(output_dir / 'paired_differences_wsi_vs_mixscon_fl.csv', index=False)
    reference.to_csv(output_dir / 'wsi_operational_reference.csv', index=False)
    config = {'replicates': replicates, 'seed': seed, 'models': models,
              'sampling': 'within-WSI patch resampling; fixed patients/WSIs; identical indices across models',
              'reference': 'predominant evaluated patch label; ties excluded; not independent pathology reference',
              'interval': '2.5th-97.5th percentiles conditional on the observed WSI set; not patient-cluster confidence intervals',
              'wsis': len(reference), 'excluded_ties': int(reference['is_tie'].sum())}
    (output_dir / 'bootstrap_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--models', nargs='+', default=['resnet50', 'vit16', 'ding', 'alsubaie', 'cellomaps', 'mixscon_ce', 'mixscon_fl'])
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    print(analyze(args.prediction_dir, args.output_dir, args.models, args.replicates, args.seed))

if __name__ == '__main__':
    main()
