
import json
import os
from datetime import date, datetime, time, timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable

from dateutil import tz
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import requests

TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.1-flash-lite-preview")
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview")


def _model_candidates(primary_model, fallback_models):
    candidates = [primary_model, *fallback_models]
    return [model for model in candidates if model]


def _load_vertex_client(genai, location_override=None):
    project = (
        os.getenv("VERTEX_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
    )
    location = location_override or (
        os.getenv("VERTEX_LOCATION")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or os.getenv("GCP_LOCATION")
        or os.getenv("GCP_REGION")
        or "global"
    )

    if project:
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.getenv("VERTEX_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    return None

def load_genai_client(env_path=None):
    """Load .env and initialise a google-genai Client using Vertex AI.

    Returns (client, True) on success, (None, False) on failure.
    """
    try:
        from google import genai  # noqa: F811
    except ImportError:
        return None, False

    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    try:
        client = _load_vertex_client(genai)
        if client is None:
            return None, False
        return client, True
    except Exception as exc:
        print(f"Vertex AI Client init failed: {exc}")
        return None, False


def _generate_images_with_fallback(client, prompt, model_candidates):
    from google.genai import types

    last_error = None
    for model in model_candidates:
        try:
            return client.models.generate_images(
                model=model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/png",
                ),
            )
        except Exception as exc:
            last_error = exc
            print(f"Image generation failed on {model}: {exc}")
    if last_error and "NOT_FOUND" in str(last_error).upper():
        try:
            from google import genai

            fallback_client = _load_vertex_client(genai, location_override="global")
            if fallback_client is not None and fallback_client is not client:
                for model in model_candidates:
                    try:
                        return fallback_client.models.generate_images(
                            model=model,
                            prompt=prompt,
                            config=types.GenerateImagesConfig(
                                number_of_images=1,
                                output_mime_type="image/png",
                            ),
                        )
                    except Exception as exc:
                        last_error = exc
                        print(f"Image generation failed on global/{model}: {exc}")
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return None


def _generate_content_with_fallback(client, prompt, model_candidates):
    last_error = None
    for model in model_candidates:
        try:
            return client.models.generate_content(model=model, contents=prompt)
        except Exception as exc:
            last_error = exc
            print(f"Text generation failed on {model}: {exc}")
    if last_error and "NOT_FOUND" in str(last_error).upper():
        try:
            from google import genai

            fallback_client = _load_vertex_client(genai, location_override="global")
            if fallback_client is not None and fallback_client is not client:
                for model in model_candidates:
                    try:
                        return fallback_client.models.generate_content(model=model, contents=prompt)
                    except Exception as exc:
                        last_error = exc
                        print(f"Text generation failed on global/{model}: {exc}")
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return None


def generate_hotspot_area_image(client, location_description, context_info, cache_dir="outputs/ai_generated"):
    """Generate an AI image depicting a typical accident-prone area.

    Uses Imagen via google-genai.  Caches the result so repeated notebook
    runs do not re-generate.
    """
    from google.genai import types

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "_- " else "" for c in location_description)[:60].strip().replace(" ", "_")
    cache_path = cache_root / f"hotspot_{safe_name}.png"
    if cache_path.exists():
        return cache_path

    prompt = (
        f"A highly detailed, photorealistic drone-shot view of a dangerous traffic hotspot in {location_description}. "
        f"{context_info} "
        "The scene must capture the specific structural flaws that cause frequent collisions: complex multi-lane intersections, "
        "confusing signage, or tightly packed highway merge lanes. Traffic is heavy and dense. "
        "Style: cinematic documentary photography, 8k resolution, dramatic golden hour or overcast lighting setting a tense mood."
    )

    try:
        response = _generate_images_with_fallback(
            client,
            prompt,
            _model_candidates(
                IMAGE_MODEL,
                [
                    "gemini-3.1-flash-image",
                    "gemini-2.5-flash-image",
                    "imagen-4.0-generate-001",
                ],
            ),
        )
        if response.generated_images:
            response.generated_images[0].image.save(location=str(cache_path))
            return cache_path
    except Exception as exc:
        print(f"Image generation failed for '{location_description}': {exc}")
    return None


def generate_severity_image(client, severity_level, sample_description, cache_dir="outputs/ai_generated"):
    """Generate an AI image depicting an accident at a given severity level."""
    from google.genai import types

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"severity_{severity_level}.png"
    if cache_path.exists():
        return cache_path

    severity_descriptions = {
        1: "a very minor fender-bender with no visible damage, slight paint scratches, cars pulled to the side",
        2: "a moderate traffic incident with minor vehicle damage, a dented bumper, slow traffic backup",
        3: "a serious multi-vehicle collision with significant vehicle damage, emergency responders on scene, road partially blocked",
        4: "a major highway accident with severely damaged vehicles, fire trucks and ambulances, full road closure, debris scattered",
    }
    severity_context = severity_descriptions.get(severity_level, severity_descriptions[2])

    prompt = (
        f"A gritty, photorealistic scene of {severity_context}. "
        f"The visual details should reflect this authentic police report constraint: '{sample_description[:200]}'. "
        "Focus on the realistic physics of the vehicular damage, the surrounding environment, and the appropriate scale of emergency response "
        "(flashing lights, tow trucks, or heavy rescue equipment depending on the severity). "
        "Style: Pulitzer Prize-winning photojournalism style, 8k resolution, cinematic lighting emphasizing the aftermath. "
        "CRITICAL: Do NOT show any graphic injuries, human victims, or blood."
    )

    try:
        response = _generate_images_with_fallback(
            client,
            prompt,
            _model_candidates(
                IMAGE_MODEL,
                [
                    "gemini-3.1-flash-image",
                    "gemini-2.5-flash-image",
                    "imagen-4.0-generate-001",
                ],
            ),
        )
        if response.generated_images:
            response.generated_images[0].image.save(location=str(cache_path))
            return cache_path
    except Exception as exc:
        print(f"Image generation failed for severity {severity_level}: {exc}")
    return None


def predict_severity_from_descriptions(client, descriptions, batch_size=25):
    """Use Gemini to predict severity (1-4) from accident descriptions.

    Returns a list of predicted severity integers.
    """
    predictions = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i : i + batch_size]
        numbered = "\n".join(f"{j+1}. {desc[:300]}" for j, desc in enumerate(batch))
        prompt = (
            "You are an expert traffic accident analyst. For each accident description below, "
            "predict a severity level from 1 to 4 where:\n"
            "  1 = Minor: little to no traffic impact\n"
            "  2 = Moderate: some traffic delay\n"
            "  3 = Serious: significant traffic impact, possible injuries\n"
            "  4 = Critical: major accident, road closure, severe injuries likely\n\n"
            "Return ONLY a JSON array of integers (one per description), e.g. [2, 3, 1, 4, ...].\n"
            "Do not include any other text.\n\n"
            f"Descriptions:\n{numbered}"
        )
        try:
            response = _generate_content_with_fallback(
                client,
                prompt,
                _model_candidates(
                    TEXT_MODEL,
                    ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite"],
                ),
            )
            text = response.text.strip()
            # Extract JSON array from response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                batch_predictions = json.loads(text[start:end])
                batch_predictions = [max(1, min(4, int(p))) for p in batch_predictions]
                predictions.extend(batch_predictions)
            else:
                predictions.extend([2] * len(batch))  # fallback
        except Exception as exc:
            print(f"Severity prediction batch {i // batch_size} failed: {exc}")
            predictions.extend([2] * len(batch))
    return predictions[:len(descriptions)]


def build_severity_correlation_analysis(ai_predictions, actual_severities):
    """Analyse agreement between AI-predicted and dataset severity levels.

    Returns a dict with correlation, confusion matrix, and bias metrics.
    """
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    ai_arr = np.array(ai_predictions)
    actual_arr = np.array(actual_severities)
    mask = ~(np.isnan(ai_arr.astype(float)) | np.isnan(actual_arr.astype(float)))
    ai_arr = ai_arr[mask].astype(int)
    actual_arr = actual_arr[mask].astype(int)

    correlation = float(np.corrcoef(ai_arr, actual_arr)[0, 1]) if len(ai_arr) > 1 else np.nan

    kappa = float(cohen_kappa_score(actual_arr, ai_arr)) if len(ai_arr) > 1 else np.nan

    exact_match = float(np.mean(ai_arr == actual_arr))

    within_one = float(np.mean(np.abs(ai_arr - actual_arr) <= 1))

    bias = float(np.mean(ai_arr - actual_arr))

    cm = confusion_matrix(actual_arr, ai_arr, labels=[1, 2, 3, 4])

    # Per-level bias
    level_bias = {}
    for level in [1, 2, 3, 4]:
        mask_level = actual_arr == level
        if mask_level.sum() > 0:
            level_bias[level] = float(np.mean(ai_arr[mask_level] - actual_arr[mask_level]))
        else:
            level_bias[level] = np.nan

    return {
        "correlation": correlation,
        "cohen_kappa": kappa,
        "exact_match_rate": exact_match,
        "within_one_rate": within_one,
        "mean_bias": bias,
        "level_bias": level_bias,
        "confusion_matrix": cm,
        "n_samples": int(len(ai_arr)),
    }


