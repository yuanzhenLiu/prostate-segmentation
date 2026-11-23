# GitHub release checklist

- Confirm `git status --short` contains only source code, configuration, documentation, and the synthetic example.
- Confirm no real image, mask, video, split spreadsheet, patient identifier, absolute local path, checkpoint, report, log, or derived visualization is staged.
- Run all command-line `--help` checks and the synthetic inference smoke test described in the README.
- Add repository ownership, contact, citation, and acknowledgments only when they are known.
- Choose a license after checking institutional policy and data agreements.
- Inspect the staged patch with `git diff --cached` before pushing.
