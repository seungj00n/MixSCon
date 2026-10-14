# MixSCon

MixSCon classifies six lung adenocarcinoma growth patterns using patient-aware mixup and supervised contrastive learning.

Code for *MixSCon: Manifold mix up-based supervised contrastive learning for robust histopathological pattern classification*.

## Environment

Use Python 3.11, PyTorch 2.0.1 and torchvision 0.15.2.

```bash
git clone https://github.com/seungj00n/MixSCon.git
cd MixSCon
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

For CPU, replace `cu118` with `cpu`. Training downloads ImageNet weights unless cached. WSI extraction and overlays also require `python -m pip install -r requirements-tiling.txt`.

## Directory structure

```text
train_mixscon.py        Patient-aware CE/FL training
train_class_mixup.py    Class-level mixup reference
evaluate.py            Tile evaluation and ROC curves
mixscon/ baselines/     Model and training code
preprocessing/         Tile extraction and split preparation
analysis/              WSI bootstrap and plots
ablation/              Six-component ablation training and external evaluation
configs/ splits/       Settings and manifest templates
data/ weights/ outputs/ Local inputs and results (Git-ignored)
```

## Data preparation

- AMC images, annotations, identifiers and split membership remain private. Prepare manifests locally using authorized data.
- TCGA: all 46 TCGA-LUAD slides with masks in [CellOMaps](https://github.com/Arwa-AlRubaian/CellOMaps). Download slides from [GDC](https://portal.gdc.cancer.gov/).
- DHMC slides: [Dartmouth Lung Cancer Histology Dataset](https://bmirds.github.io/LungCancer/). Evaluation also requires the study's region annotations and selected-slide manifest, which are separate from this repository.

Labels: `0 acinar, 1 cribriform, 2 lepidic, 3 micropapillary, 4 papillary, 5 solid`.
RGB tiles are extracted at 768 x 768 and resized to 224 x 224. CellOMaps requires separately prepared cell maps, resized to 448 x 448 and reduced to 224 x 224.

```bash
python -m preprocessing.extract_tiles_tif --data-root data/tif_with_xml --output-dir data/tiles
python -m preprocessing.extract_tiles_mrxs --wsi data/slide.mrxs --tumor-xml data/tumor.xml --pattern-xml data/pattern.xml --output-dir data/tiles
```

TIFF extraction expects matching `slide.tif` and `slide.xml`. XML annotations use ASAP format and level-0 coordinates; both commands produce PNG tiles.

Fill a local copy of `splits/folds.example.csv` as `data/folds.csv`: `sample_id,patient_id,wsi_id,fold,split,label,path`. Use folds 1-5 for `train`/`val` and fold 0 for the shared `test` set. Keep identifiers consistent across folds; paths are relative to `data/tiles`.

```bash
python -m preprocessing.prepare_splits --manifest data/folds.csv --data-root data/tiles --output-dir data/cv
```

This checks patient/WSI separation and copies tiles into `Round1/train/1_acinar/` through `Round5/val/6_solid/`, plus `test/<class>/`, under `data/cv`. Templates describe the format; original AMC splits remain private.

## Training

Run from the repository root. The default trains five folds; `--folds 1` runs one. Use `--dry-run` to check inputs.

```bash
python train_mixscon.py --data-root data/cv --loss ce --output-dir outputs/train_ce
python train_mixscon.py --data-root data/cv --loss fl --output-dir outputs/train_fl
python train_class_mixup.py --data-root data/cv --output-dir outputs/train_class
python -m baselines.train --model vit16 --data-root data/cv --output-dir outputs/vit
python -m baselines.train --model alsubaie --data-root data/cv --output-dir outputs/alsubaie
python -m baselines.train --model cellomaps --data-root data/cellmap_cv --output-dir outputs/cellomaps
python -m baselines.train --model ding --folds 1 --data-root data/cv --ssl-checkpoints weights/ssl1.pth weights/ssl2.pth weights/ssl3.pth --output-dir outputs/ding
```

MixSCon defaults: AdamW, learning rate 0.001, weight decay 0.0001, batch 32, maximum 200 epochs, seed 2025. CE/FL settings are in `configs/`.

The CE/FL checkpoints use a ResNet-50 encoder, a 2048-to-128 linear projection and a six-class cosine classifier on pooled 2048-D features (temperature 0.07). Mixup operates on normalized projections. The training code initializes classifier weights with Kaiming normal initialization. The class-level reference uses an MLP classifier.

The training scripts save one CE/FL checkpoint per fold using validation cross-entropy (patience 30; minimum improvement 0.0001). Select the checkpoint to evaluate in the JSON configuration below.

Baseline learning rates (Adam): ViT 0.001, Alsubaie 0.0001, CellOMaps 0.00001, Ding 0.0001. Alsubaie uses RGB plus a downsampled/upsampled copy; Ding requires three fold-specific SSL checkpoints with ResNet-compatible keys. ResNet-50 is supported for evaluation. See `NOTICE` for method sources.

Table VII uses two-stage training: an encoder trained with Adam, followed by an MLP trained with AdamW on the frozen encoder (both learning rates 0.001; classifier weight decay 0.0001). Manifold mixup is applied after ResNet layer 3. Encoder checkpoints use training loss; classifier checkpoints use validation accuracy.

```bash
python -m ablation.run_fair_ablation_study --data-root data/cv --output-dir outputs/ablation
python -m ablation.run_fair_ablation_study_evaluate --cohort TCGA --data-root data/TCGA_tiles --model-root outputs/ablation --output-dir outputs/ablation_TCGA
```

Both commands accept `--folds 1`, `--experiments 1 2 3 4 5 6`, `--skip-tsne` and `--dry-run`. Model files are saved under `Round*/models/`; evaluation CSVs and plots under `Round*/results/`.

## Evaluation and WSI analysis

Provide compatible weights separately. Copy `configs/checkpoints.example.json` to `weights/checkpoints.json` and replace its placeholder paths. Fill `splits/tiles.example.csv` locally as `data/AMC.csv`: unique `sample_id`, `label`, `raw_path`, and `cellmap_path` for CellOMaps. Paths are relative to the manifest directory. WSI analysis also needs `patient_id,wsi_id`.

The `label` column accepts class names (e.g. `lepidic` or `3_lepidic`) or fixed IDs. TCGA uses `0,2,3,4,5`; do not use compact ImageFolder IDs `0..4`. The ablation evaluator applies `map_labels` automatically to folder names, including when Cribriform is absent.

```bash
python evaluate.py --cohort AMC --manifest data/AMC.csv --checkpoint-config weights/checkpoints.json --weights-root weights --output-dir outputs/AMC --device cuda
python -m analysis.bootstrap --prediction-dir outputs/AMC --output-dir outputs/AMC_wsi --replicates 1000 --seed 2026
```

Repeat with `TCGA` and `DHMC` in place of `AMC`, then run:

```bash
python -m analysis.plot_wsi --amc outputs/AMC_wsi --tcga outputs/TCGA_wsi --dhmc outputs/DHMC_wsi --output-dir outputs/figures
```

Evaluation and bootstrap accept `--models mixscon_ce mixscon_fl`; combined plots expect all seven models. CPU inference uses `--device cpu`.

Evaluation reports macro metrics; training summaries use weighted precision/F1. TCGA retains all six outputs: Cribriform predictions count as errors, without probability renormalization. Ding averages three softmax outputs equally.

Bootstrap resamples patches within each fixed WSI using shared indices across models. WSI predictions use mean probabilities; reference labels use the most frequent evaluated patch label, excluding ties. One WSI per patient is required. Percentile intervals describe patch-sampling variability within the observed cohort.

For Fig. 4, use the prediction CSV from `evaluate.py` or the original CSV with `filename,pred,confidence score` columns:

```bash
python -m analysis.overlay_patch_wsi --wsi data/slide.mrxs --xml data/slide.xml --predictions outputs/AMC/mixscon_fl_tile_predictions.csv --output outputs/slide_overlay.png --image-level 3 --confidence-threshold 0.7 --patch-scale 5 --coordinate-origin center
```

These are the original Fig. 4 settings. Coordinates come from original `WSI_class_x_X_y_Y.png` filenames or explicit `wsi_id,x,y` columns (level-0 pixels; `wsi_id` matches the slide basename). For tiles from `preprocessing/`, use `--coordinate-origin top-left --patch-scale 1` to show their actual footprints. The script saves the overlay PNG and settings JSON.

## Outputs and manuscript mapping

| Script | Main outputs | Manuscript |
|---|---|---|
| `preprocessing/` | PNG tiles and supplied split folders | Table I: data preparation |
| Training scripts, `configs/` | MixSCon: `best_model_round_*.pth`, `cross_validation_results.json`, training plots; baseline-specific checkpoints/metrics | Tables II-III, VIII: training/configuration |
| `evaluate.py` | `point_estimates.csv`, per-class metrics, tile predictions, ROC and confusion matrices | Tables IV-VI, Fig. 2 |
| `train_class_mixup.py`, `train_mixscon.py`, `evaluate.py` | Class/patient comparison metrics and ROC | Table IX, Fig. 3(b-d) |
| `analysis.bootstrap`, `analysis.plot_wsi` | WSI metrics, paired differences, AUC plots | Tables R1-R3, Figs. R1-R2 |
| `ablation.run_fair_ablation_study`, `ablation.run_fair_ablation_study_evaluate` | Encoder/classifier checkpoints, metrics, ROC and t-SNE | Table VII |
| `analysis.overlay_patch_wsi` | WSI overlay PNG and settings JSON | Fig. 4 |
| `evaluate.py` and Excel | Per-class AUCs and tile prediction CSVs | Fig. 3(a), Table X |

For class-reference evaluation, add a `class_mixup` checkpoint to the JSON. Fig. 3(a) and Table X were prepared in Excel using per-class AUCs and prediction CSVs. The CSVs include filenames, true/predicted classes, confidence and all six probabilities.

## License and citation

Author-written code uses the [MIT License](LICENSE); third-party terms are listed in [NOTICE](NOTICE). Data and weights are separate.

Please cite the manuscript above by Eunha Lee, Seungjoon Baek, Sangmin Lee, Ji-Su Uh, Jinseong Kim, Hee Sang Hwang, Se Jin Jang and Heounjeong Go. Publication details will be added when available.
