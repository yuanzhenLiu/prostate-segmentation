# Final Classic U-Net v2 method

## Data and split

The two valid source pairs are `image1 → mask1` and `image2 → mask2`. Pairing uses an identical filename stem. Patient and frame identifiers are parsed from `P-XXX_YYY`.

Patients are shuffled with split seed 2026 and assigned approximately 70%, 10%, and 20% to train, validation, and test. The split is made at patient level before frame sampling. The split generator verifies that each patient belongs to exactly one subset.

Only training rows are sampled. Within each `(source, patient_id)` group, rows are sorted by numeric frame identifier and stem, every fourth row is selected, and the first 180 selected rows are retained. There is no perceptual or image-similarity filter. Validation and test rows remain complete.

## Preprocessing and augmentation

Images are converted to RGB and resized to 512 × 512 with bilinear interpolation. Masks are converted to one channel, resized with nearest-neighbor interpolation, and binarized at values greater than zero. Images are scaled to `[0, 1]` and normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

Training augmentation follows the historical implementation:

- horizontal flip with probability 0.5;
- rotation with probability 0.5, uniformly sampled from −10° to +10°;
- brightness and contrast adjustment, each with probability 0.5 and factor range 0.92–1.08;
- additive Gaussian noise with probability 0.5 and standard deviation 0.01;
- Gaussian blur with probability 0.05 and radius from 0.3 to 0.8.

Vertical flipping is not implemented.

## Network and optimization

Classic U-Net uses two 3 × 3 convolution, batch-normalization, and ReLU operations in each block. Dropout2d is applied after the first ReLU in every block. Channel widths are 24, 48, 96, 192, and 384. Four transposed convolutions and skip connections form the decoder. A 1 × 1 convolution produces one logit channel.

The network is initialized from scratch. There is no pretrained encoder and no frozen stage. Optimization uses AdamW with learning rate `1e-4` and weight decay `3e-3`. The loss is BCEWithLogits plus soft Dice loss. Training uses mixed precision, batch size 8, at most 35 epochs, and early stopping after five epochs without validation Dice improvement of at least `1e-4`.

## Metrics and output scale

Dice, IoU/Jaccard, precision, recall/sensitivity, specificity, accuracy, and symmetric 95th-percentile Hausdorff distance are computed per test frame. The default probability threshold is 0.5. Empty prediction and empty target give HD95 0; if only one is empty, HD95 is undefined (`NaN`).

Labeled test evaluation operates on the resized 512 × 512 grid, so HD95 is in pixels on that grid. Deployment inference restores the probability map to the original image dimensions before thresholding. Its mask, probability array, and red overlay therefore have the original input size.
