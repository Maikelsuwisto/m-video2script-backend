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

# Helper: Convert seconds to HH:MM:SS
def seconds_to_hms(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

@app.post("/transcribe")
async def transcribe(
    video: UploadFile = File(...),
    with_timestamps: str = Form("0")  # "1" = include timestamps, "0" = no timestamps
):
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
                entry["start"] = seconds_to_hms(seg.start)
                entry["end"] = seconds_to_hms(seg.end)
            transcription.append(entry)

        # Optional: generate audio URL
        audio_url = None

        return {"transcription": transcription, "audio_url": audio_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process transcription: {str(e)}")

    finally:
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
