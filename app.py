from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import shutil
import tempfile
from pathlib import Path

from detector import AIImageDetector
from metadata import analyze_metadata
from image_analysis import (
    analyze_frequency,
    analyze_noise_consistency,
    detect_face_anomalies,
)

app = FastAPI(title="AI Image Detector API")


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Initialize AI detector
# --------------------------------------------------

try:
    detector = AIImageDetector()
    detector_status = "loaded"
except Exception as e:
    detector = None
    detector_status = f"failed: {str(e)}"


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}


def normalize_score(value):
    """
    Convert a score to 0-100.

    If the input is between 0 and 1,
    assume it is a probability.

    If it is already between 0 and 100,
    keep it as it is.
    """

    try:
        value = float(value)

        if 0 <= value <= 1:
            return value * 100

        if 0 <= value <= 100:
            return value

        return 0

    except (TypeError, ValueError):
        return 0


def get_ai_detection_score(ai_result):
    """
    Convert detector output into an AI probability from 0-100.

    Assumes:
        prediction = "ai"
        confidence = confidence of predicted class

    If prediction is "real", the AI probability becomes
    100 - confidence.
    """

    if not isinstance(ai_result, dict):
        return 0

    prediction = str(
        ai_result.get("prediction", "")
    ).lower()

    confidence = normalize_score(
        ai_result.get("confidence", 0)
    )

    if prediction == "ai":
        return confidence

    if prediction == "real":
        return 100 - confidence

    return 0


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "AI Image Detector API is running",
        "detector_status": detector_status,
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "detector_loaded": detector is not None,
    }


# --------------------------------------------------
# Image detection
# --------------------------------------------------

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):

    # ----------------------------------------------
    # Validate filename
    # ----------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing"
        )

    # ----------------------------------------------
    # Validate content type
    # ----------------------------------------------

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image format. "
                "Allowed formats: JPG, PNG, WEBP, GIF, BMP"
            ),
        )

    # ----------------------------------------------
    # Check detector
    # ----------------------------------------------

    if detector is None:
        raise HTTPException(
            status_code=500,
            detail="AI detector model failed to initialize"
        )

    # ----------------------------------------------
    # Create temporary directory
    # ----------------------------------------------

    temp_dir = tempfile.mkdtemp()

    # Use only the filename, preventing path traversal
    safe_filename = Path(file.filename).name

    temp_path = os.path.join(
        temp_dir,
        safe_filename
    )

    try:

        # ------------------------------------------
        # Save uploaded file
        # ------------------------------------------

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ------------------------------------------
        # Basic file validation
        # ------------------------------------------

        file_size = os.path.getsize(temp_path)

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        # ------------------------------------------
        # AI detection
        # ------------------------------------------

        try:
            ai_detection = detector.predict(temp_path)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"AI detector error: {str(e)}"
            )

        # ------------------------------------------
        # Prepare results
        # ------------------------------------------

        results = {
            "filename": safe_filename,
            "file_size": file_size,
            "ai_detection": ai_detection,
        }

        # ------------------------------------------
        # Metadata analysis
        # ------------------------------------------

        try:

            results["metadata"] = analyze_metadata(
                temp_path
            )

        except Exception as e:

            results["metadata"] = {
                "error": str(e)
            }

        # ------------------------------------------
        # Frequency analysis
        # ------------------------------------------

        try:

            results["frequency_analysis"] = analyze_frequency(
                temp_path
            )

        except Exception as e:

            results["frequency_analysis"] = {
                "error": str(e)
            }

        # ------------------------------------------
        # Noise analysis
        # ------------------------------------------

        try:

            results["noise_analysis"] = analyze_noise_consistency(
                temp_path
            )

        except Exception as e:

            results["noise_analysis"] = {
                "error": str(e)
            }

        # ------------------------------------------
        # Face anomaly analysis
        # ------------------------------------------

        try:

            results["face_analysis"] = detect_face_anomalies(
                temp_path
            )

        except Exception as e:

            results["face_analysis"] = {
                "error": str(e)
            }

        # ==========================================
        # Calculate overall AI score
        # ==========================================

        scores = []
        weights = []

        # ------------------------------------------
        # AI model
        # ------------------------------------------

        ai_score = get_ai_detection_score(
            results["ai_detection"]
        )

        scores.append(ai_score)
        weights.append(0.40)

        # ------------------------------------------
        # Metadata
        # ------------------------------------------

        metadata = results.get("metadata", {})

        if isinstance(metadata, dict):
            if "score" in metadata:

                metadata_score = normalize_score(
                    metadata["score"]
                )

                scores.append(metadata_score)
                weights.append(0.20)

        # ------------------------------------------
        # Frequency
        # ------------------------------------------

        frequency = results.get(
            "frequency_analysis",
            {}
        )

        if isinstance(frequency, dict):

            if "ai_likelihood" in frequency:

                frequency_score = normalize_score(
                    frequency["ai_likelihood"]
                )

                scores.append(frequency_score)
                weights.append(0.20)

        # ------------------------------------------
        # Noise
        # ------------------------------------------

        noise = results.get(
            "noise_analysis",
            {}
        )

        if isinstance(noise, dict):

            if "ai_score" in noise:

                noise_score = normalize_score(
                    noise["ai_score"]
                )

                scores.append(noise_score)
                weights.append(0.20)

        # ------------------------------------------
        # Weighted average
        # ------------------------------------------

        if scores and weights:

            total_score = sum(
                score * weight
                for score, weight
                in zip(scores, weights)
            )

            weight_sum = sum(weights)

            overall_score = (
                total_score / weight_sum
            )

        else:

            overall_score = 50

        overall_score = round(
            max(0, min(overall_score, 100)),
            2
        )

        # ------------------------------------------
        # Verdict
        # ------------------------------------------

        if overall_score >= 60:

            verdict = "AI-Generated"

        elif overall_score <= 40:

            verdict = "Likely Real"

        else:

            verdict = "Uncertain"

        # ------------------------------------------
        # Final response
        # ------------------------------------------

        results["overall_ai_score"] = overall_score
        results["verdict"] = verdict

        return results

    finally:

        # ------------------------------------------
        # Cleanup
        # ------------------------------------------

        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

        except Exception:
            pass


# --------------------------------------------------
# Run server
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )