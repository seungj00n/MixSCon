"""Within-WSI patch resampling; patient/WSI membership is held fixed."""
from __future__ import annotations
from typing import Dict, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
CLASS_NAMES = ['Acinar', 'Cribriform', 'Lepidic', 'Micropapillary', 'Papillary', 'Solid']
PROBABILITY_COLUMNS = ['prob_' + name.lower() for name in CLASS_NAMES]
MODEL_ORDER = ['resnet50', 'vit16', 'ding', 'alsubaie', 'cellomaps', 'mixscon_ce', 'mixscon_fl']
DISPLAY_NAMES = dict(zip(MODEL_ORDER, ['ResNet-50', 'ViT-16', 'Ding et al.', 'Alsubaie et al.', 'CellOMaps', 'MixSCon (CE)', 'MixSCon (FL)']))
METRICS = ['auc', 'precision', 'accuracy', 'f1']
def metric_values(
    y_true: np.ndarray, probabilities: np.ndarray, evaluation_labels: Sequence[int]
) -> Dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    auc_values = []
    for class_id in evaluation_labels:
        binary = (y_true == class_id).astype(np.int64)
        if np.unique(binary).size == 2:
            auc_values.append(roc_auc_score(binary, probabilities[:, class_id]))
    return {
        "auc": float(np.mean(auc_values)) if auc_values else float("nan"),
        "precision": float(
            precision_score(
                y_true, predictions, labels=list(evaluation_labels), average="macro", zero_division=0
            )
        ),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(
            f1_score(
                y_true, predictions, labels=list(evaluation_labels), average="macro", zero_division=0
            )
        ),
    }

def bootstrap_indices(
    manifest: pd.DataFrame, replicates: int, seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: Dict[str, np.ndarray] = {}
    for wsi_id, group in manifest.groupby("wsi_id", sort=True):
        indices = group.index.to_numpy(np.int64)
        result[wsi_id] = rng.choice(indices, size=(replicates, len(indices)), replace=True)
    return result

def dominant_reference(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wsi_id, group in manifest.groupby("wsi_id", sort=True):
        counts = np.bincount(group["label"].to_numpy(np.int64), minlength=len(CLASS_NAMES))
        maximum = int(counts.max())
        winners = np.flatnonzero(counts == maximum)
        rows.append(
            {
                "patient_id": group["patient_id"].iloc[0],
                "wsi_id": wsi_id,
                "n_patches": len(group),
                "reference_label": int(winners[0]) if len(winners) == 1 else -1,
                "reference_class": (
                    CLASS_NAMES[int(winners[0])]
                    if len(winners) == 1
                    else "+".join(CLASS_NAMES[index] for index in winners)
                ),
                "is_tie": len(winners) != 1,
                "dominant_count": maximum,
                "dominant_fraction": maximum / len(group),
                **{f"n_{CLASS_NAMES[i].lower()}": int(counts[i]) for i in range(6)},
            }
        )
    return pd.DataFrame(rows)

def percentile_summary(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {key: float("nan") for key in ["mean", "std", "median", "q025", "q25", "q75", "q975"]}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "q025": float(np.quantile(finite, 0.025)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "q975": float(np.quantile(finite, 0.975)),
    }

def summarize(metrics: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    for model, group in metrics.groupby("model", sort=False):
        for metric in METRICS:
            stats = percentile_summary(group[metric])
            rows.append(
                {
                    "level": level,
                    "model": model,
                    "display_name": DISPLAY_NAMES[model],
                    "metric": metric,
                    **stats,
                    "iqr": stats["q75"] - stats["q25"],
                    "ci_width": stats["q975"] - stats["q025"],
                }
            )
    return pd.DataFrame(rows)

def paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    proposed = metrics.loc[metrics["model"].eq("mixscon_fl")].sort_values("replicate")
    for comparator in MODEL_ORDER:
        if comparator == "mixscon_fl":
            continue
        other = metrics.loc[metrics["model"].eq(comparator)].sort_values("replicate")
        if len(proposed) != len(other):
            continue
        for metric in METRICS:
            difference = proposed[metric].to_numpy(float) - other[metric].to_numpy(float)
            stats = percentile_summary(difference)
            rows.append(
                {
                    "proposed": "mixscon_fl",
                    "comparator": comparator,
                    "comparison": f"MixSCon (FL) - {DISPLAY_NAMES[comparator]}",
                    "metric": metric,
                    **stats,
                    "probability_difference_gt_0": float(np.mean(difference > 0)),
                    "ci_excludes_zero": bool(stats["q025"] > 0 or stats["q975"] < 0),
                }
            )
    return pd.DataFrame(rows)

def point_wsi_metrics(
    probabilities: np.ndarray, manifest: pd.DataFrame, reference: pd.DataFrame
) -> Dict[str, float]:
    means = pd.DataFrame(probabilities, columns=PROBABILITY_COLUMNS)
    means["wsi_id"] = manifest["wsi_id"]
    means = means.groupby("wsi_id", sort=True)[PROBABILITY_COLUMNS].mean()
    evaluable = reference.loc[~reference["is_tie"]].sort_values("wsi_id")
    labels = sorted(evaluable["reference_label"].unique().tolist())
    return metric_values(
        evaluable["reference_label"].to_numpy(np.int64),
        means.loc[evaluable["wsi_id"]].to_numpy(float),
        labels,
    )

def bootstrap_model(
    model_name: str,
    probabilities: np.ndarray,
    manifest: pd.DataFrame,
    samples: Mapping[str, np.ndarray],
    reference: pd.DataFrame,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = manifest["label"].to_numpy(np.int64)
    patch_eval_labels = sorted(np.unique(labels).tolist())
    combined = np.concatenate([samples[wsi] for wsi in sorted(samples)], axis=1)
    patch_rows = []
    for replicate in range(replicates):
        selected = combined[replicate]
        patch_rows.append(
            {
                "model": model_name,
                "replicate": replicate + 1,
                **metric_values(labels[selected], probabilities[selected], patch_eval_labels),
            }
        )

    reference_index = reference.set_index("wsi_id")
    wsi_probability_arrays: Dict[str, np.ndarray] = {}
    wsi_summary_rows = []
    for wsi_id in sorted(samples):
        averaged = probabilities[samples[wsi_id]].mean(axis=1)
        wsi_probability_arrays[wsi_id] = averaged
        predictions = averaged.argmax(axis=1)
        ref = reference_index.loc[wsi_id]
        ref_label = int(ref["reference_label"])
        if ref_label >= 0:
            ref_stats = percentile_summary(averaged[:, ref_label])
            correctness = predictions == ref_label
            modal_prediction = int(np.bincount(predictions, minlength=6).argmax())
            stability = float(np.mean(predictions == modal_prediction))
            accuracy = float(np.mean(correctness))
        else:
            ref_stats = percentile_summary([])
            modal_prediction = int(np.bincount(predictions, minlength=6).argmax())
            stability = float(np.mean(predictions == modal_prediction))
            accuracy = float("nan")
        max_stats = percentile_summary(averaged.max(axis=1))
        wsi_summary_rows.append(
            {
                "model": model_name,
                "display_name": DISPLAY_NAMES[model_name],
                "patient_id": ref["patient_id"],
                "wsi_id": wsi_id,
                "n_patches": int(ref["n_patches"]),
                "reference_class": ref["reference_class"],
                "reference_is_tie": bool(ref["is_tie"]),
                "modal_predicted_class": CLASS_NAMES[modal_prediction],
                "prediction_stability": stability,
                "bootstrap_accuracy": accuracy,
                "reference_probability_median": ref_stats["median"],
                "reference_probability_q025": ref_stats["q025"],
                "reference_probability_q975": ref_stats["q975"],
                "maximum_probability_median": max_stats["median"],
                "maximum_probability_q025": max_stats["q025"],
                "maximum_probability_q975": max_stats["q975"],
            }
        )

    evaluable = reference.loc[~reference["is_tie"]].sort_values("wsi_id")
    wsi_true = evaluable["reference_label"].to_numpy(np.int64)
    wsi_eval_labels = sorted(np.unique(wsi_true).tolist())
    wsi_rows = []
    for replicate in range(replicates):
        wsi_probabilities = np.stack(
            [wsi_probability_arrays[wsi_id][replicate] for wsi_id in evaluable["wsi_id"]], axis=0
        )
        wsi_rows.append(
            {
                "model": model_name,
                "replicate": replicate + 1,
                **metric_values(wsi_true, wsi_probabilities, wsi_eval_labels),
            }
        )
    return pd.DataFrame(patch_rows), pd.DataFrame(wsi_rows), pd.DataFrame(wsi_summary_rows)
