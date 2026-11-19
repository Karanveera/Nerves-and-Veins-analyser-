"""Vein-detection API (host this on Render / Railway / Hugging Face Spaces / any server).
Run locally:  uvicorn api:app --host 0.0.0.0 --port 8000
The HTML page POSTs a photo to /predict and gets back the marked image.
"""
import base64
import cv2, numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from vein_detector import process

app = FastAPI(title="Vein Detector API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# (allow_origins=["*"] lets index.html opened from your laptop call this API.
#  Once you host index.html on a fixed address, replace "*" with that address.)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), sensitivity: float = Form(0.8),
                  hand: str = Form("auto"), min_length: int = Form(30),
                  max_veins: int = Form(12)):
    data = await file.read()
    if len(data) > 10_000_000:
        raise HTTPException(413, "Image too large (max 10 MB)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a valid image")
    hand = None if hand == "none" else hand
    _, marked, n = process(img, sensitivity, min_length, hand=hand, max_veins=max_veins)
    ok, buf = cv2.imencode(".png", marked)
    return {"count": int(n), "image": "data:image/png;base64," + base64.b64encode(buf).decode()}
