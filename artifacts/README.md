# VAYU-NET — Runtime Artifacts Directory

This directory manages the runtime artifact specifications and download manifests for VAYU-NET.

## Overview
Model checkpoints and heavy runtime assets are decoupled from the Git repository to keep commits lightweight. The application retrieves its production weights on-demand using:

```bash
# Check if all runtime artifacts exist and verify their SHA-256 integrity
python scripts/setup/download_runtime_artifacts.py --check

# Dry-run download verification
python scripts/setup/download_runtime_artifacts.py --dry-run

# Download any missing runtime models
python scripts/setup/download_runtime_artifacts.py
```

## Structure
- `manifest.json`: Definitive index of model checkpoints, metadata configurations, and demo assets with verified SHA-256 checksums.
- `models/`: Local staging directory where runtime weights (`*.pt`) reside.
