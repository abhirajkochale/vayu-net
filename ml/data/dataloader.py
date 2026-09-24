"""
VAYU-NET — PyTorch DataLoader Factory
Constructs deterministic DataLoaders for TRAIN, VALIDATION, and TEST splits.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .vayu_dataset import VayuSatelliteDataset, DEFAULT_SAMPLE_INDEX, DEFAULT_NORM_STATS

def seed_worker(worker_id):
    """Ensures deterministic random states inside DataLoader worker processes"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_vayu_dataloaders(sample_index_csv=DEFAULT_SAMPLE_INDEX,
                         norm_stats_path=DEFAULT_NORM_STATS,
                         batch_size=8,
                         num_workers=0,
                         pin_memory=True,
                         seed=42):
    """
    Factory function returning ready-to-use PyTorch DataLoaders for all three splits.

    Args:
        sample_index_csv: Path to authoritative sample index.
        norm_stats_path: Path to training-only normalization JSON.
        batch_size: Mini-batch size.
        num_workers: Subprocess workers for data loading (0 recommended on Windows).
        pin_memory: Pin memory for faster GPU tensor transfer.
        seed: Random seed for deterministic reproducibility.

    Returns:
        dict with keys: 'train', 'val', 'test', 'datasets'
    """
    # Deterministic PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    # 1. Instantiate datasets
    train_dataset = VayuSatelliteDataset(
        sample_index_csv=sample_index_csv,
        split="TRAIN",
        norm_stats_path=norm_stats_path,
        normalize=True
    )

    val_dataset = VayuSatelliteDataset(
        sample_index_csv=sample_index_csv,
        split="VALIDATION",
        norm_stats_path=norm_stats_path,
        normalize=True
    )

    test_dataset = VayuSatelliteDataset(
        sample_index_csv=sample_index_csv,
        split="TEST",
        norm_stats_path=norm_stats_path,
        normalize=True
    )

    # 2. Instantiate DataLoaders
    # Use pin_memory only if CUDA is available or explicitly requested without warning
    use_pin = pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        worker_init_fn=seed_worker
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        worker_init_fn=seed_worker
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "datasets": {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }
    }
