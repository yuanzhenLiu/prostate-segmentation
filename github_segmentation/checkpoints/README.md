# Checkpoints

Trained weights are not included in this public repository. Place a local checkpoint outside the repository, or copy it into this directory for local use. Files ending in `.pt`, `.pth`, and `.ckpt` are ignored by Git.

The final v2 checkpoint must contain either a raw PyTorch state dictionary or a dictionary with a `model` state dictionary. The historical checkpoint also contains `model_name`, `image_size`, and `best_dice`. It was created with the model settings in `configs/unet_v2.json`: Classic U-Net, base channels 24, dropout 0.15, three input channels, and one output channel.

Example local path:

```text
checkpoints/unet_best.pt
```

Example prediction:

```powershell
python scripts/predict.py --input examples/input/synthetic_ultrasound.png --checkpoint checkpoints/unet_best.pt --output-dir outputs/example --device auto
```

No public download URL is currently available. Train the model locally or supply an authorized private checkpoint.
