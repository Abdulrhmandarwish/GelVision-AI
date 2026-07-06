"""Preprocessing module for gel electrophoresis images.

Provides image loading, grayscale conversion, contrast enhancement (CLAHE),
Gaussian denoising, normalization, and U-Net tensor preparation utilities.
"""

from __future__ import annotations

from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image


def preprocess_gel(
    image_input: Union[str, np.ndarray, Image.Image],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess a gel electrophoresis image for downstream analysis.

    The pipeline applies the following sequential transformations:
      1. Grayscale conversion (if the input is colour).
      2. CLAHE (Contrast Limited Adaptive Histogram Equalization) to balance
         uneven illumination across the gel.
      3. Gaussian blur (3×3 kernel) to suppress high-frequency noise.
      4. Min-max normalization to the [0, 255] intensity range.

    Args:
        image_input: Input gel image.  Accepted types:
            * ``str`` — file-system path to an image file.
            * ``numpy.ndarray`` — raw pixel array (BGR colour or grayscale).
            * ``PIL.Image.Image`` — Pillow image object.

    Returns:
        A 3-tuple of ``numpy.ndarray`` images (all uint8, single-channel):
            * **original_gray** — the input converted to grayscale.
            * **clahe_image** — after CLAHE contrast enhancement.
            * **final_preprocessed** — after Gaussian blur + normalization.

    Raises:
        FileNotFoundError: If *image_input* is a path that cannot be read.
        TypeError: If *image_input* is not one of the supported types.
    """
    # --- Load / convert to grayscale ---
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Image not found or unreadable at path: {image_input}"
            )
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input.copy()
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("L"))
    else:
        raise TypeError(
            "Unsupported image input type. Must be a file path (str), "
            "numpy array, or PIL Image. "
            f"Received: {type(image_input).__name__}"
        )

    # --- CLAHE to fix uneven lighting ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # --- Gaussian blur for denoising ---
    img_denoised = cv2.GaussianBlur(img_clahe, (3, 3), 0)

    # --- Normalize to 0-255 range ---
    img_norm = cv2.normalize(img_denoised, None, 0, 255, cv2.NORM_MINMAX)

    return img, img_clahe, img_norm


def prepare_unet_input(
    crop: np.ndarray,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Prepare a grayscale lane crop for U-Net inference.

    The crop is resized to 512×512, normalised to [0, 1], and wrapped in a
    PyTorch tensor with batch and channel dimensions suitable for a
    single-channel U-Net (shape ``(1, 1, 512, 512)``).

    Args:
        crop: 2-D grayscale ``numpy.ndarray`` representing a single lane crop.
            Expected dtype is ``uint8`` with values in [0, 255].
        device: Target device for the returned tensor (e.g. ``"cpu"`` or
            ``"cuda"``).  Accepts both string identifiers and
            ``torch.device`` objects.

    Returns:
        A ``torch.Tensor`` of shape ``(1, 1, 512, 512)`` with ``float32``
        dtype, moved to the requested *device*.

    Raises:
        ValueError: If *crop* is empty or has unexpected dimensionality.
    """
    if crop is None or crop.size == 0:
        raise ValueError("Received an empty crop array — cannot prepare U-Net input.")
    if crop.ndim != 2:
        raise ValueError(
            f"Expected a 2-D grayscale crop, but got shape {crop.shape}."
        )

    # Resize to the U-Net's expected spatial resolution
    resized = cv2.resize(crop, (512, 512))

    # Normalize pixel intensities to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Build (N, C, H, W) tensor and move to device
    tensor = (
        torch.tensor(normalized, dtype=torch.float32)
        .unsqueeze(0)   # add channel dim  → (1, 512, 512)
        .unsqueeze(0)   # add batch  dim   → (1, 1, 512, 512)
        .to(device)
    )
    return tensor
