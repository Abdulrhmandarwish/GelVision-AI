"""Analysis pipeline module for GelVision AI.

Orchestrates the full gel electrophoresis analysis workflow:
YOLO lane detection → U-Net band segmentation → calibration → results.

Provides standalone model-loading functions designed for use with
``@st.cache_resource`` in the Streamlit front-end.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.signal import find_peaks
from ultralytics import YOLO

import segmentation_models_pytorch as smp

from src.preprocess import preprocess_gel
from src.calibration import calibrate_ladder


# ---------------------------------------------------------------------------
# Standalone model loaders (designed for @st.cache_resource)
# ---------------------------------------------------------------------------

def load_yolo_model(model_path: str = "models/best.pt") -> YOLO:
    """Load a YOLOv8 model from disk.

    Args:
        model_path: Path to the ``.pt`` weights file.

    Returns:
        An initialised ``ultralytics.YOLO`` model ready for inference.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found at: {model_path}")
    return YOLO(model_path)


def load_unet_model(
    model_path: str = "models/unet_best.pth",
    device: Optional[torch.device] = None,
) -> smp.Unet:
    """Load a U-Net (ResNet-34 encoder) band-segmentation model.

    Args:
        model_path: Path to the ``.pth`` state-dict file.
        device: Target device.  Defaults to CUDA if available, else CPU.

    Returns:
        A ``segmentation_models_pytorch.Unet`` model in eval mode on *device*.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"U-Net model not found at: {model_path}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation="sigmoid",
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class GelAnalysisPipeline:
    """End-to-end gel electrophoresis analysis pipeline.

    Accepts **pre-loaded** models so that callers (e.g. the Streamlit app)
    can cache them independently with ``@st.cache_resource``.
    """

    def __init__(
        self,
        yolo_model: YOLO,
        unet_model: smp.Unet,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialise the pipeline with pre-loaded models.

        Args:
            yolo_model: A loaded ``YOLO`` object for lane detection.
            unet_model: A loaded ``smp.Unet`` in eval mode for band segmentation.
            device: Torch device for U-Net inference.
        """
        self.yolo_model = yolo_model
        self.unet_model = unet_model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        conf_threshold: float = 0.3,
        band_threshold: float = 0.3,
        min_band_distance: int = 10,
        ladder_lane_number: Optional[int] = None,
        ladder_type: str = "1kb_plus",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """Run the complete GelVision AI pipeline.

        Args:
            image_input: Gel image as a file path, numpy array, or PIL Image.
            conf_threshold: YOLO confidence threshold for lane detection.
            band_threshold: Minimum peak height for ``find_peaks``.
            min_band_distance: Minimum pixel distance between band peaks.
            ladder_lane_number: 1-indexed lane number to use as ladder.
                ``None`` means auto-detect via YOLO class label.
            ladder_type: Key into ``LADDER_TYPES`` (e.g. ``"1kb_plus"``).
            progress_callback: Optional ``(fraction, message)`` callback.

        Returns:
            Dictionary with keys:

            * ``img_preprocessed`` — original grayscale image.
            * ``img_annotated`` — annotated BGR image with bounding boxes.
            * ``lanes`` — list of lane dicts (each has crop, mask, bands, …).
            * ``results_df`` — ``pandas.DataFrame`` of band results.
            * ``ladder_lane_number`` — lane index used as ladder.
            * ``calibration_params`` — ``(slope, intercept)`` tuple.
            * ``calibration_r2`` — R² of the calibration fit.
        """
        # ── Step 1: Preprocess ───────────────────────────────────────
        if progress_callback:
            progress_callback(0.1, "Preprocessing gel image…")
        original_gray, img_clahe, img_norm = preprocess_gel(image_input)

        # Prepare colour image for annotation drawing (BGR)
        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                img_annotated = image_input.copy()
            else:
                img_annotated = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif isinstance(image_input, Image.Image):
            img_annotated = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:  # file path
            img_annotated = cv2.imread(str(image_input))
            if img_annotated is None:
                img_annotated = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)

        # ── Step 2: YOLOv8 Lane Detection ────────────────────────────
        if progress_callback:
            progress_callback(0.3, "Detecting lanes with YOLOv8…")

        results = self.yolo_model.predict(
            source=img_annotated, conf=conf_threshold, verbose=False
        )
        boxes = results[0].boxes

        # ── Step 3: Sort lanes left → right ──────────────────────────
        if progress_callback:
            progress_callback(0.4, "Sorting and cropping lanes…")

        detected_boxes: List[Dict] = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label_idx = int(box.cls[0])
            label_name = results[0].names[label_idx]
            conf = float(box.conf[0])
            detected_boxes.append({
                "bbox": (x1, y1, x2, y2),
                "label": label_name,
                "confidence": conf,
            })

        detected_boxes = sorted(detected_boxes, key=lambda b: b["bbox"][0])

        # Assign 1-indexed lane numbers
        lanes: List[Dict[str, Any]] = []
        for idx, item in enumerate(detected_boxes):
            lanes.append({
                "number": idx + 1,
                "label": item["label"],
                "confidence": item["confidence"],
                "bbox": item["bbox"],
            })

        # ── Step 4: U-Net Band Segmentation ──────────────────────────
        if progress_callback:
            progress_callback(0.6, "Segmenting bands with U-Net…")

        for lane in lanes:
            x1, y1, x2, y2 = lane["bbox"]
            # Crop from the preprocessed (normalised) grayscale image
            crop = img_norm[y1:y2, x1:x2]

            # Resize to 512×512 for U-Net
            lane_resized = cv2.resize(crop, (512, 512))
            lane_float = lane_resized / 255.0

            # Build tensor → inference
            lane_tensor = (
                torch.tensor(lane_float, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                pred = self.unet_model(lane_tensor)

            # Binary mask (threshold at 0.5)
            pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            lane["crop"] = lane_resized
            lane["mask"] = pred_mask

        # ── Step 5: Band Feature Extraction ──────────────────────────
        if progress_callback:
            progress_callback(0.8, "Extracting band features…")

        for lane in lanes:
            # 1-D horizontal profile of the mask
            profile = lane["mask"].mean(axis=1)

            peaks, _ = find_peaks(
                profile, height=band_threshold, distance=min_band_distance
            )

            bands: List[Dict[str, Any]] = []
            for b_idx, peak in enumerate(peaks):
                y_min = max(0, peak - 5)
                y_max = min(512, peak + 5)
                mean_intensity = float(lane["crop"][y_min:y_max, :].mean())

                bands.append({
                    "band_number": b_idx + 1,
                    "centroid_y": float(peak),
                    "intensity": mean_intensity,
                    "profile_height": float(profile[peak]),
                })
            lane["bands"] = bands

        # ── Step 6: Ladder Identification & Calibration ──────────────
        if progress_callback:
            progress_callback(0.9, "Calibrating band sizes…")

        # Determine the ladder lane
        ladder_lane: Optional[Dict] = None
        if ladder_lane_number is not None and ladder_lane_number > 0:
            for l in lanes:
                if l["number"] == ladder_lane_number:
                    ladder_lane = l
                    break
        else:
            # Auto-detect from YOLO label
            for l in lanes:
                if l["label"] == "ladder":
                    ladder_lane = l
                    break
            # Fallback: assume last lane is the ladder
            if ladder_lane is None and len(lanes) > 0:
                ladder_lane = lanes[-1]

        # Calibration
        r2_score = 0.0
        calibration_func: Callable[[float], int] = lambda y: 0
        slope, intercept = 0.0, 0.0

        if ladder_lane is not None:
            ladder_lane["label"] = "ladder"
            ladder_positions = [b["centroid_y"] for b in ladder_lane["bands"]]
            calibration_func, (slope, intercept), r2_score, _ = calibrate_ladder(
                ladder_positions, ladder_type=ladder_type
            )

        # ── Step 7: Annotate Image & Build Results ───────────────────
        results_records: List[Dict] = []
        for lane in lanes:
            x1, y1, x2, y2 = lane["bbox"]
            is_ladder = (
                lane["number"] == (ladder_lane["number"] if ladder_lane else -1)
            )

            # Colour-coded bounding boxes: RED for ladder, GREEN for lanes
            color = (0, 0, 255) if is_ladder else (0, 255, 0)
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
            label_text = (
                f"Lane {lane['number']} "
                f"({'Ladder' if is_ladder else 'Lane'})"
            )
            cv2.putText(
                img_annotated, label_text,
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

            # Estimate band sizes
            for band in lane["bands"]:
                est_size = calibration_func(band["centroid_y"])
                band["estimated_size_bp"] = est_size

                if not is_ladder:
                    results_records.append({
                        "Lane": lane["number"],
                        "Label": lane["label"],
                        "Lane_Confidence": lane["confidence"],
                        "Band": band["band_number"],
                        "Position": band["centroid_y"],
                        "Estimated_Size_bp": est_size,
                        "Intensity": band["intensity"],
                    })

        df_results = pd.DataFrame(results_records)

        # ── Step 8: Done ─────────────────────────────────────────────
        if progress_callback:
            progress_callback(1.0, "Analysis complete!")

        return {
            "img_preprocessed": original_gray,
            "img_annotated": img_annotated,
            "lanes": lanes,
            "results_df": df_results,
            "ladder_lane_number": ladder_lane["number"] if ladder_lane else None,
            "calibration_params": (slope, intercept),
            "calibration_r2": r2_score,
        }
