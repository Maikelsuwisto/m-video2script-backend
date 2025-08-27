import os
import shutil
import uuid
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

# -----------------------
# Setup FastAPI
# -----------------------
app = FastAPI(title="MS-Video2Script Backend")

# CORS (allow frontend calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://web-production-3f9e.up.railway.app"],  # adjust if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React build folder
app.mount("/static", StaticFiles(directory="build/static"), name="static")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    file_path = os.path.join("build", full_path)
    if not os.path.exists(file_path):
        file_path = os.path.join("build", "index.html")
    response = FileResponse(file_path)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response

# -----------------------
# Whisper transcription
# -----------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe(
    video: UploadFile = File(...),
    with_timestamps: str = Form("0")
):
    unique_filename = f"{uuid.uuid4()}_{video.filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        segments, info = model.transcribe(save_path, beam_size=5)

        transcription = []
        for seg in segments:
            entry = {"text": seg.text}
            if with_timestamps == "1":
                hours = int(seg.start // 3600)
                minutes = int((seg.start % 3600) // 60)
                secs = int(seg.start % 60)
                entry["start"] = f"{hours:02d}:{minutes:02d}:{secs:02d}"

                hours = int(seg.end // 3600)
                minutes = int((seg.end % 3600) // 60)
                secs = int(seg.end % 60)
                entry["end"] = f"{hours:02d}:{minutes:02d}:{secs:02d}"
            transcription.append(entry)

        audio_url = None
        return {"transcription": transcription, "audio_url": audio_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

# -----------------------
# Debug exception handler
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

# Health check
@app.get("/")
def health():
    return {"message": "MS-Video2Script API running ✅"}
