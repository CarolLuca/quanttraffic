from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from typing import Tuple

import numpy as np


@lru_cache(maxsize=1)
def _has_nvidia_gpu() -> Tuple[bool, str]:
    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return True, "Detected NVIDIA GPU via nvidia-smi"
        if probe.returncode == 0:
            return False, "nvidia-smi returned no GPU devices"
        stderr = probe.stderr.strip() or "nvidia-smi command failed"
        return False, stderr
    except Exception as exc:
        return False, f"nvidia-smi unavailable: {exc}"


@lru_cache(maxsize=1)
def _xgboost_cuda_ready() -> Tuple[bool, str]:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:
        return False, f"xgboost import failed: {exc}"

    has_gpu, gpu_msg = _has_nvidia_gpu()
    if not has_gpu:
        return False, gpu_msg

    try:
        x = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.5, 1.0]], dtype=float)
        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        model = xgb.XGBRegressor(
            n_estimators=1,
            max_depth=1,
            learning_rate=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        model.fit(x, y)
        return True, "XGBoost CUDA backend validated"
    except Exception as exc:
        return False, f"xgboost CUDA init failed: {exc}"


def use_gpu_acceleration(enable_gpu: bool = True) -> bool:
    if not enable_gpu:
        return False
    if os.getenv("AAD_DISABLE_GPU", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    ready, _ = _xgboost_cuda_ready()
    return ready


def gpu_status_note(enable_gpu: bool = True) -> str:
    if not enable_gpu:
        return "GPU disabled by configuration"
    if os.getenv("AAD_DISABLE_GPU", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "GPU disabled by AAD_DISABLE_GPU environment variable"
    ready, message = _xgboost_cuda_ready()
    return message if ready else f"GPU acceleration unavailable ({message})"
