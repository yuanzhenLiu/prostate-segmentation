# Outputs, reproducibility, and public-code changes

## Generated outputs

Training writes `checkpoints/unet_best.pt` and `history_unet.json` under the selected output directory. Test evaluation writes `metrics_per_image.csv`, `metrics_summary.csv`, `prediction_overlays/`, and `visualizations/`. Inference writes a binary mask, red ROI overlay, and NumPy probability array for every input image. Curve plotting writes loss and Dice PNG files plus CSV tables.

All these directories may contain patient-derived information and are ignored by Git.

## Historical experiment boundary

The verified final history contains 16 epochs. Validation Dice peaked at 0.825984 in epoch 11 and early stopping occurred in epoch 16. The private test summary was Dice 0.823264, IoU 0.722785, precision 0.850261, recall/sensitivity 0.848139, and HD95 29.198682 pixels.

These values describe one private historical run. The original code did not fully set Python, NumPy, and PyTorch seeds. The public training script keeps the seed unset by default to describe that run accurately and offers `--seed` for future experiments. Different hardware, PyTorch versions, data ordering, and nondeterministic CUDA operations can change results.

## Necessary public-code fixes

Two bookkeeping defects were corrected without changing the model, optimizer, loss, augmentations, or gradients:

1. The earlier evaluator read only index zero from each batch and therefore required batch size 1. The public evaluator loops over every item, so larger evaluation batches cannot silently discard images.
2. The earlier training display divided accumulated training values by the full dataset size even though `drop_last=True` may omit a final incomplete batch. The public display divides by the actual processed count. This can slightly change displayed training curves, but it does not affect parameter updates, validation scores, early stopping, or checkpoint selection.

The public model definition was verified by strict loading of the private v2 checkpoint. The checkpoint itself is intentionally excluded.

## Privacy and release boundary

Do not commit real images, masks, videos, split spreadsheets, patient/frame identifiers, checkpoints, per-image metrics, logs, Word reports, or patient-derived visualizations. Review `git status` and `git diff --cached` before every public push. No license is included; choose one only after confirming institutional and data-governance requirements.
