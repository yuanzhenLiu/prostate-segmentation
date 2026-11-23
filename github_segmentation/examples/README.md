# Synthetic example

`input/synthetic_ultrasound.png` is procedurally generated and contains no patient data. Recreate it with:

```powershell
python scripts/create_synthetic_example.py --output examples/input/synthetic_ultrasound.png
```

If a compatible local checkpoint is available, run:

```powershell
python scripts/predict.py --input examples/input/synthetic_ultrasound.png --checkpoint checkpoints/unet_best.pt --output-dir outputs/example --device auto
```

The generated prediction outputs are ignored by Git.
