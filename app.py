from fastapi import FastAPI

app = FastAPI(title="MS-Video2Script Backend - Minimal Test")

@app.get("/")
def root():
    return {"message": "Hello, Railway! ✅"}
