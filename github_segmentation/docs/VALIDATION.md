# Validation record

Validation was performed on 2026-11-23 without running full training.

- All Python files passed bytecode compilation.
- Every script entry point completed `--help`, confirming imports and argument parsing.
- The public Classic U-Net definition strictly loaded the private final v2 state dictionary.
- GPU inference on a synthetic 640 × 480 image produced a 640 × 480 binary mask, overlay, and probability map. The mask contained only 0 and 255, and every overlay background pixel was identical to the input.
- Batch evaluation used five synthetic labeled images with batch size 2 and wrote five metric rows, including the final one-item batch.
- A synthetic ten-patient split produced 7/1/2 train/validation/test patients with zero cross-split leakage and zero unmatched files.
- The historical JSON training history produced both loss and Dice curves, a full CSV history, and a best-validation CSV row.
- A text scan found no project drive paths, usernames, old project-directory references, or environment-specific Python paths in public text files.

Full training and exact metric reproduction were intentionally not run. The real dataset, split table, checkpoint, and patient-derived outputs remain outside this repository.
