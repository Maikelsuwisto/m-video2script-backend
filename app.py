# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
import traceback
from whisper_utils import transcribe_video  # Make sure this function exists

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

        # Call your transcription function
        transcription_text = transcribe_video(save_path, include_timestamps=(with_timestamps=="1"))

        # Optional: generate audio URL (if your function supports it)
        audio_url = None

        return {"transcription": transcription_text, "audio_url": audio_url}

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
