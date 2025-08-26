from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

# Lazy import Whisper and pydub to avoid startup delays
whisper = None

app = FastAPI(title="MS-Video2Script Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads folder exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_whisper_model(model_name="tiny"):
    """Load Whisper model only when needed to save memory/startup time."""
    global whisper
    if whisper is None:
        import whisper
    return whisper.load_model(model_name)

def transcribe_video(file_path, include_timestamps=False):
    """Transcribe video using Whisper tiny model"""
    model = get_whisper_model("tiny")
    result = model.transcribe(file_path)
    text = result.get("text", "")
    # Add timestamps formatting if needed
    if include_timestamps:
        # Placeholder: you can format segments with timestamps
        text = f"[timestamps enabled]\n{text}"
    return text

@app.post("/transcribe")
async def transcribe(
    video: UploadFile = File(...),
    with_timestamps: str = Form("0")
):
    unique_filename = f"{uuid.uuid4()}_{video.filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Save uploaded video temporarily
        with open(save_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # Transcribe
        try:
            transcription_text = transcribe_video(
                save_path,
                include_timestamps=(with_timestamps == "1")
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        return {"transcription": transcription_text, "audio_url": None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save/process file: {str(e)}")

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

@app.get("/")
def root():
    return {"message": "MS-Video2Script API is running ✅"}

@app.get("/shutdown")
def shutdown():
    """Forcefully stops the FastAPI server"""
    os._exit(0)
