"""
semantic.py -- factors derived from labels / split metadata (Stage B, no pixels).

Signature for meta factors: fn(split_name, labels) -> (N,) array, or None if the factor
is undefined for that split (the probe then drops those rows).
"""
import numpy as np

from ..context import parse_corrupt_split
from ..registry import factor

# CIFAR-10 class order: airplane automobile bird cat deer dog frog horse ship truck
CIFAR10_VEHICLE = {0, 1, 8, 9}

CORRUPTION_FAMILY = {
    "gaussian_noise": "noise", "shot_noise": "noise", "impulse_noise": "noise",
    "defocus_blur": "blur", "glass_blur": "blur", "motion_blur": "blur", "zoom_blur": "blur",
    "snow": "weather", "frost": "weather", "fog": "weather",
    "brightness": "digital", "contrast": "digital", "elastic_transform": "digital",
    "pixelate": "digital", "jpeg_compression": "digital",
}
FAMILY_INDEX = {"clean": 0, "noise": 1, "blur": 2, "weather": 3, "digital": 4}
CORRUPTION_INDEX = {c: i + 1 for i, c in enumerate(CORRUPTION_FAMILY)}   # 0 = clean


@factor("class", kind="categorical", pool="clean", source="meta")
def class_factor(split, labels):
    """Ground-truth class label (the primary semantic factor)."""
    return labels.astype(int)


@factor("coarse_animal_vehicle", kind="categorical", pool="clean", n_classes=2, source="meta")
def coarse_factor(split, labels):
    """CIFAR-10 coarse split: 0 = animal, 1 = vehicle (airplane/automobile/ship/truck)."""
    return np.isin(labels, list(CIFAR10_VEHICLE)).astype(int)


@factor("corruption_family", kind="categorical", pool="corrupt", n_classes=5, source="meta")
def family_factor(split, labels):
    """0 clean | 1 noise | 2 blur | 3 weather | 4 digital (CIFAR-10-C families)."""
    pc = parse_corrupt_split(split)
    if pc is None:
        return np.zeros(len(labels), dtype=int) if split == "test" else None
    return np.full(len(labels), FAMILY_INDEX[CORRUPTION_FAMILY[pc[0]]], dtype=int)


@factor("corruption_type", kind="categorical", pool="corrupt", n_classes=16, source="meta")
def type_factor(split, labels):
    """0 clean, 1..15 = CIFAR-10-C corruption index."""
    pc = parse_corrupt_split(split)
    if pc is None:
        return np.zeros(len(labels), dtype=int) if split == "test" else None
    return np.full(len(labels), CORRUPTION_INDEX[pc[0]], dtype=int)


@factor("severity", kind="continuous", pool="corrupt", source="meta")
def severity_factor(split, labels):
    """Corruption severity 0 (clean) .. 5, as a regression target."""
    pc = parse_corrupt_split(split)
    if pc is None:
        return np.zeros(len(labels), dtype=np.float32) if split == "test" else None
    return np.full(len(labels), float(pc[1]), dtype=np.float32)
