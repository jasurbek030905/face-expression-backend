from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from fer import FER
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.post("/detect-emotion")
async def detect_emotion(file: UploadFile = File(...)):
    global detector

    if detector is None:
        detector = FER(mtcnn=False)

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (640, 480))

    results = detector.detect_emotions(frame)

    if not results:
        return {"emotion": "no face", "confidence": 0}

    emotions = results[0]["emotions"]
    emotion = max(emotions, key=emotions.get)
    score = emotions[emotion]

    return {
        "emotion": emotion,
        "confidence": float(score)
    }