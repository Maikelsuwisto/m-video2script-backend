# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import shutil
import os
import uuid
import traceback

app = FastAPI(title="MS-Video2Script Backend")

# Enable CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (you can restrict later)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads folder exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load tiny model once (fastest for Railway free tier)
model = WhisperModel("tiny", device="cpu", compute_type="int8")


@app.post("/transcribe")
async def transcribe(
    video: UploadFile = File(...),
    with_timestamps: str = Form("0")  # "1" = include timestamps, "0" = no timestamps
):
    # Generate unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4()}_{video.filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Save uploaded video temporarily
        with open(save_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # Run transcription
        segments, info = model.transcribe(save_path, beam_size=5)

        transcription = []
        for seg in segments:
            entry = {"text": seg.text}
            if with_timestamps == "1":
                entry["start"] = f"{seg.start:.2f}"
                entry["end"] = f"{seg.end:.2f}"
            transcription.append(entry)

        # Optional: generate audio URL (not implemented yet)
        audio_url = None

        return {"transcription": transcription, "audio_url": audio_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process transcription: {str(e)}")

    finally:
        # Delete the uploaded file after processing
        if os.path.exists(save_path):
            os.remove(save_path)


# -----------------------
# Debug Exception Handler
# -----------------------
@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc()
        },
    )


# -----------------------
# Health check route
# -----------------------
@app.get("/")
def health():
    return {"message": "MS-Video2Script API is running ✅"}
