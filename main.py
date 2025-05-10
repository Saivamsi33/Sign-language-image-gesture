import os
import time
import logging
from threading import Thread, Event
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import shutil

# Environment-aware configuration
IS_CLOUD = os.getenv("CLOUD_ENV", "false").lower() == "true"
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads" if IS_CLOUD else "static/uploads")
RESULT_DIR = os.getenv("RESULT_DIR", "/tmp/results" if IS_CLOUD else "static/results")
CLEANUP_INTERVAL = 300  # 5 minutes in seconds

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Static files setup
static_dir = "/tmp" if IS_CLOUD else "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory="templates")

# Smart Cleaner Service
class FileCleaner:
    def __init__(self):
        self.active = Event()
        self.thread = Thread(target=self.run, daemon=True)
        self.last_activity = 0
        self.thread.start()
        logger.info("File cleaner initialized")

    def run(self):
        while True:
            if self.active.is_set():
                self.clean_files()
                # Sleep only if no recent activity
                if time.time() - self.last_activity > CLEANUP_INTERVAL:
                    time.sleep(60)  # Normal interval
                else:
                    time.sleep(10)  # Active processing interval
            else:
                time.sleep(5)  # Low-power mode

    def clean_files(self):
        now = time.time()
        for folder in [UPLOAD_DIR, RESULT_DIR]:
            if not os.path.exists(folder):
                continue
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                try:
                    if os.path.isfile(filepath) and (now - os.path.getmtime(filepath)) > CLEANUP_INTERVAL:
                        os.remove(filepath)
                        logger.info(f"Deleted {filepath}")
                except Exception as e:
                    logger.error(f"Error deleting {filepath}: {str(e)}")

    def notify_activity(self):
        self.last_activity = time.time()
        if not self.active.is_set():
            self.active.set()
            logger.info("Cleaner activated")

# Initialize cleaner
cleaner = FileCleaner()

@app.on_event("shutdown")
def shutdown():
    cleaner.active.set()  # Ensure thread exits cleanly

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    try:
        # Notify cleaner of new activity
        cleaner.notify_activity()

        # Process upload
        timestamp = int(time.time())
        safe_name = f"{timestamp}_{file.filename.replace(' ', '_')}"
        upload_path = os.path.join(UPLOAD_DIR, safe_name)
        result_path = os.path.join(RESULT_DIR, safe_name)

        # Save file
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Process image (YOLO detection)
        image = cv2.imread(upload_path)
        results = YOLO("best.pt").predict(image)
        
        # Draw bounding boxes (simplified example)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imwrite(result_path, image)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "uploaded": True,
            "file_path": f"/static/results/{safe_name}",
            "prediction": f"Sign {int(results[0].boxes.cls[0])}",
            "confidence": f"{float(results[0].boxes.conf[0])*100:.0f}"
        })

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(500, "Processing error")
    finally:
        await file.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)