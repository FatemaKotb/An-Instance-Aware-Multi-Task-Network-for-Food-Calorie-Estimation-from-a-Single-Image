# clip_finetune

Self-contained CLIP fine-tuning (Approach B): **freeze CLIP image encoder + train a tiny classifier** for cuisine-specific dish labels.

This mirrors the style of the existing `PredictionHeadPretrainedCLIP` (OpenCLIP, crop+mask support), but trains a linear head
for your Egyptian dish classes (folder-per-class dataset).

## Dataset layout

```
DATA_ROOT/
  koshari/
    *.jpg
  ful_medames/
    *.jpg
  ...
```

Delete any unwanted images directly from the folders (watermarks, etc.).

## Run

Open `notebooks/01_train_eval.ipynb`, set `DATA_ROOT`, then run all cells.

Outputs:
- `artifacts/splits/` train/val/test lists
- `artifacts/ckpts/egypt_clip_linear.pt` checkpoint
- `artifacts/reports/metrics.json` metrics including zero-shot baseline
