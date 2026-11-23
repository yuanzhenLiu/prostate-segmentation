# Classic U-Net for Binary Segmentation of Video Frames

This repository contains the final **Classic U-Net v2** pipeline from a binary ROI segmentation study on images extracted from videos. It covers patient-level splitting, training-frame subsampling, GPU training, test evaluation, single-image or folder inference, and learning-curve plotting.

The medical images, masks, patient-level split tables, trained weights, and patient-derived visualizations are private and are not included.

## Final method

- Input/label pairs: `image1` with `mask1`, and `image2` with `mask2`.
- Patient-level train/validation/test split of approximately 7:1:2. A patient cannot occur in more than one split.
- Training-only video-frame sampling: group by source and patient, sort by frame number, take every fourth frame, and keep at most the first 180 sampled frames per group. Validation and test sets are not sampled.
- Classic U-Net trained from scratch: four encoder levels, four decoder levels, base width 24, dropout 0.15, one binary output channel.
- Images and masks are resized to 512 × 512 for training and test evaluation.
- AdamW, learning rate `1e-4`, weight decay `3e-3`, BCE + soft Dice loss, batch size 8, at most 35 epochs, and early-stopping patience 5.
- Augmentation: horizontal flip, ±10° rotation, mild brightness and contrast changes, Gaussian noise, and Gaussian blur. No vertical flip is used.
- Inference restores probabilities to the original image size, thresholds them at 0.5, and overlays the ROI in red `(255, 0, 0)` with alpha 0.45.

The exact settings are in [`configs/unet_v2.json`](configs/unet_v2.json). See [`docs/METHOD.md`](docs/METHOD.md) for implementation details.

## Repository layout

```text
configs/                       final v2 JSON configuration
src/ultrasound_segmentation/  model, data, loss, metrics, checkpoint, visualization
scripts/                       split, train, evaluate, predict, and plot entry points
docs/                          method, outputs, reproducibility, and release notes
examples/                      synthetic non-patient example and instructions
checkpoints/                   local weight placement instructions only
```

## Installation

Python 3.10 or newer is recommended. Create and activate an environment, then install a CUDA-compatible PyTorch build for the target GPU using the official PyTorch installation instructions. PyTorch is intentionally absent from `requirements.txt` so an existing working GPU installation is not replaced.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print((torch.ones(1, device='cuda') + 1).item())"
```

The last command must complete before GPU training. Inference also supports `--device cpu`.

## Data layout and naming

```text
DATA_ROOT/
├── image1/
├── mask1/
├── image2/
└── mask2/
```

An image and its mask must have the same filename stem. Filenames follow `P-XXX_YYY`, where `XXX` identifies the patient and `YYY` identifies the frame.

Create a private patient-level split table:

```powershell
python scripts/prepare_split.py --data-root "D:\path\to\date" --output "D:\private\dataset_split.xlsx" --seed 2026
```

The table contains patient identifiers and absolute file paths and must not be committed. To preserve an existing split after moving data:

```powershell
python scripts/relocate_split.py --input "D:\private\old_split.xlsx" --data-root "E:\new\date" --output "E:\private\dataset_split.xlsx"
```

## Train

Training requires CUDA:

```powershell
python scripts/train.py --config configs/unet_v2.json --split-table "D:\private\dataset_split.xlsx" --output-dir "D:\private\unet_v2_run"
```

If Windows multiprocessing causes a worker error, add `--num-workers 0`. If GPU memory is insufficient, add `--batch-size 4`; this changes the run configuration. The original historical run did not set a complete global random seed. Use `--seed 2026` for future, more controlled runs, but do not expect exact reproduction of the historical metrics.

## Evaluate a labeled test set

```powershell
python scripts/evaluate.py --config configs/unet_v2.json --split-table "D:\private\dataset_split.xlsx" --checkpoint "D:\private\unet_best.pt" --output-dir "D:\private\evaluation" --batch-size 4 --num-visuals 30 --device cuda
```

Evaluation processes every item in every batch. It writes per-image and summary CSV files, red prediction overlays, and image/green-ground-truth/red-prediction comparisons. Metrics are calculated at 512 × 512. HD95 is therefore measured in pixels on that grid. Summary values are the mean and sample standard deviation across frames; video frames are not independent patients.

## Predict images without labels

Single image:

```powershell
python scripts/predict.py --input "D:\images\example.png" --checkpoint "D:\private\unet_best.pt" --output-dir "D:\predictions" --device cuda
```

Whole directory, recursively:

```powershell
python scripts/predict.py --input "D:\images" --checkpoint "D:\private\unet_best.pt" --output-dir "D:\predictions" --device auto
```

Each input produces `*_mask.png`, `*_overlay.png`, and `*_probability.npy`. The mask, overlay, and probability map have the original input dimensions. New unlabeled images can be segmented but cannot produce true Dice or other accuracy metrics.

## Plot training curves

```powershell
python scripts/plot_curves.py --history "D:\private\unet_v2_run\history_unet.json" --output-dir "D:\private\unet_v2_run\curves"
```

Training is shown as a solid line and validation as a dashed line of the same color.

## Verified historical result

The final private v2 experiment used 67 patients and 23,106 paired frames. The split contained 47/7/13 patients and 16,288/2,161/4,657 frames in train/validation/test. Training stopped after epoch 16; the best validation Dice was 0.825984 at epoch 11.

Frame-level test means were: Dice 0.823264, IoU 0.722785, precision 0.850261, recall/sensitivity 0.848139, and HD95 29.198682 pixels. These values describe the private historical experiment. No public data or checkpoint is bundled, and they are not a promise of exact reproduction.

## Reproducibility and common issues

- The historical run did not set Python, NumPy, and PyTorch global seeds. The code exposes `--seed` for future runs.
- Data, split tables, and weights must be supplied locally. See [`checkpoints/README.md`](checkpoints/README.md).
- `CUDA is unavailable`: install a PyTorch build compatible with the GPU and driver, then run the CUDA verification command above.
- `no kernel image is available`: the installed PyTorch CUDA build does not support the GPU architecture.
- Missing image/mask errors: verify matching stems and the four required directories, or relocate the split table.
- A direct `train.py ...` command fails in PowerShell because Python scripts are not shell commands; run `python scripts/train.py ...`.

Further limitations and two bookkeeping fixes made for the public code are documented in [`docs/OUTPUTS_AND_REPRODUCIBILITY.md`](docs/OUTPUTS_AND_REPRODUCIBILITY.md).

## License and citation

No license is included. Before publishing, choose a license that is compatible with your institution, data agreements, and intended reuse. No author, institution, paper, citation, or public checkpoint URL is asserted by this repository.
