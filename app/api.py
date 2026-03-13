"""
FastAPI application – exposes REST endpoints for bin status, alerts, and
trash classification.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from loguru import logger

from app.config import settings
from app.models.bin_status import BinReading
from app.models.classification import ClassificationResult
from app.sensors.ultrasonic import SimulatedUltrasonicSensor
from app.services.camera_service import camera
from app.services.classifier_service import classifier
from app.services.monitor_service import MonitorService
from app.utils.logging import setup_logging

monitor = MonitorService()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    setup_logging()
    logger.info(f"Starting {settings.app_name} ...")

    # Register simulated bins (replace with real sensors in production)
    for bin_id, fill_pct in [("BIN-A", 0.0), ("BIN-B", 50.0), ("BIN-C", 75.0)]:
        monitor.register_sensor(
            SimulatedUltrasonicSensor(
                bin_id=bin_id,
                capacity_cm=settings.bin_capacity_cm,
                initial_fill_percent=fill_pct,
            )
        )

    monitor.start()

    # Load the trash classification model (non-blocking if missing)
    classifier.load_model()

    # Initialise camera (non-blocking if no camera attached)
    camera.initialise()

    yield
    monitor.stop()
    camera.release()
    logger.info(f"{settings.app_name} shut down.")


app = FastAPI(
    title=settings.app_name,
    description="IoT Smart Trash Bin monitoring system",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["System"])
async def root() -> dict[str, Any]:
    return {
        "app": settings.app_name,
        "version": "1.0.0",
        "classifier_ready": classifier.is_ready,
        "camera_ready": camera.is_ready,
        "endpoints": {
            "health": "/health",
            "all_bins": "/bins",
            "single_bin": "/bins/{bin_id}",
            "classify": "POST /classify",
            "capture_and_classify": "POST /capture-and-classify",
            "camera_feed": "/camera/feed",
            "camera_stream": "/camera/stream",
            "docs": "/docs",
        },
    }


@app.get("/health", tags=["System"])
async def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.get("/bins", response_model=dict[str, Any], tags=["Bins"])
async def list_bins() -> dict[str, Any]:
    """Return the latest reading for every registered bin."""
    readings = monitor.get_latest_readings()
    return {
        bin_id: {
            "fill_level_percent": r.fill_level_percent,
            "fill_status": r.fill_status,
            "distance_cm": r.distance_cm,
            "capacity_cm": r.capacity_cm,
            "timestamp": r.timestamp.isoformat(),
        }
        for bin_id, r in readings.items()
    }


@app.get("/bins/{bin_id}", response_model=dict[str, Any], tags=["Bins"])
async def get_bin(bin_id: str) -> dict[str, Any]:
    """Return the latest reading for a specific bin."""
    readings = monitor.get_latest_readings()
    if bin_id not in readings:
        raise HTTPException(status_code=404, detail=f"Bin '{bin_id}' not found.")
    r: BinReading = readings[bin_id]
    return {
        "bin_id": r.bin_id,
        "fill_level_percent": r.fill_level_percent,
        "fill_status": r.fill_status,
        "distance_cm": r.distance_cm,
        "capacity_cm": r.capacity_cm,
        "timestamp": r.timestamp.isoformat(),
    }



@app.post("/classify", response_model=ClassificationResult, tags=["Classification"])
async def classify_image(file: UploadFile = File(...)) -> ClassificationResult:
    """
    Upload an image of trash and get its predicted category.

    Categories (TrashNet): cardboard, glass, metal, paper, plastic, trash.
    """
    if not classifier.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Classification model not loaded. "
                "Train it first: python -m app.training.train"
            ),
        )

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got content type: '{content_type}'",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = classifier.classify(image_bytes)
    return result


@app.post("/capture-and-classify", response_model=ClassificationResult, tags=["Classification"])
async def capture_and_classify() -> ClassificationResult:
    """
    Capture an image from the attached camera and classify the trash.

    No file upload needed – the Pi camera takes the photo automatically.
    Categories (TrashNet): cardboard, glass, metal, paper, plastic, trash.
    """
    if not camera.is_ready:
        raise HTTPException(
            status_code=503,
            detail="No camera available. Attach a camera and restart the app.",
        )

    if not classifier.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Classification model not loaded. "
                "Train it first: python -m app.training.train"
            ),
        )

    # Capture image from camera
    image_bytes = camera.capture()

    # Save the captured image for reference
    saved_path = camera.save_capture(image_bytes)
    logger.info(f"Captured image saved to '{saved_path}'")

    # Classify
    result = classifier.classify(image_bytes)
    return result


@app.get("/camera/stream", tags=["Camera"])
async def camera_stream():
    """
    Raw MJPEG stream from the attached camera.

    Use this URL directly in an <img> tag or VLC/ffplay.
    """
    if not camera.is_ready:
        raise HTTPException(
            status_code=503,
            detail="No camera available. Attach a camera and restart the app.",
        )

    return StreamingResponse(
        camera.stream_frames(fps=10),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/camera/feed", response_class=HTMLResponse, tags=["Camera"])
async def camera_feed():
    """
    View the live camera feed in your browser.

    Open http://<pi-ip>:8001/camera/feed
    """
    if not camera.is_ready:
        raise HTTPException(
            status_code=503,
            detail="No camera available. Attach a camera and restart the app.",
        )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{settings.app_name} – Camera Feed</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                background: #1a1a2e; color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
                display: flex; flex-direction: column;
                align-items: center; padding: 20px;
                min-height: 100vh;
            }}
            h1 {{ color: #00d4aa; margin-bottom: 15px; font-size: 1.6em; }}
            .container {{
                display: flex; flex-wrap: wrap; justify-content: center;
                gap: 20px; width: 100%; max-width: 1100px;
            }}
            .video-panel {{
                position: relative; flex: 1; min-width: 320px;
            }}
            .video-panel img {{
                width: 100%; border: 3px solid #00d4aa;
                border-radius: 10px; display: block;
            }}
            .overlay {{
                position: absolute; top: 12px; left: 12px;
                background: rgba(0,0,0,0.7); padding: 8px 14px;
                border-radius: 6px; font-size: 1.3em; font-weight: bold;
                display: none; z-index: 10;
            }}
            .overlay.show {{ display: block; }}
            .side-panel {{
                flex: 0 0 320px; display: flex; flex-direction: column; gap: 12px;
            }}
            .controls {{ display: flex; gap: 10px; }}
            button {{
                background: #00d4aa; color: #1a1a2e; border: none;
                padding: 14px 20px; font-size: 15px; font-weight: bold;
                border-radius: 8px; cursor: pointer; flex: 1;
                transition: background 0.2s;
            }}
            button:hover {{ background: #00b894; }}
            button:disabled {{ background: #555; cursor: not-allowed; }}
            #autoBtn.active {{ background: #e74c3c; }}
            #autoBtn.active:hover {{ background: #c0392b; }}
            .result-card {{
                background: #16213e; border-radius: 10px;
                padding: 18px; min-height: 120px;
            }}
            .result-card h2 {{
                font-size: 1.1em; color: #00d4aa; margin-bottom: 10px;
            }}
            .category-label {{
                font-size: 2em; font-weight: bold; margin: 5px 0;
            }}
            .confidence {{
                font-size: 1.2em; color: #aaa; margin-bottom: 12px;
            }}
            .bar-chart {{ display: flex; flex-direction: column; gap: 6px; }}
            .bar-row {{
                display: flex; align-items: center; gap: 8px; font-size: 0.85em;
            }}
            .bar-label {{ width: 75px; text-align: right; }}
            .bar-track {{
                flex: 1; height: 18px; background: #0f3460;
                border-radius: 4px; overflow: hidden;
            }}
            .bar-fill {{
                height: 100%; border-radius: 4px;
                transition: width 0.4s ease;
            }}
            .bar-pct {{ width: 45px; text-align: right; font-size: 0.8em; color: #aaa; }}
            .history {{
                background: #16213e; border-radius: 10px; padding: 14px;
                max-height: 200px; overflow-y: auto;
            }}
            .history h2 {{ font-size: 1em; color: #00d4aa; margin-bottom: 8px; }}
            .history-item {{
                display: flex; justify-content: space-between;
                padding: 4px 0; border-bottom: 1px solid #0f3460;
                font-size: 0.85em;
            }}
            .status {{ font-size: 0.9em; color: #aaa; text-align: center; margin-top: 5px; }}
        </style>
    </head>
    <body>
        <h1>📷 {settings.app_name} – Live Feed</h1>
        <div class="container">
            <div class="video-panel">
                <div class="overlay" id="overlay"></div>
                <img src="/camera/stream" alt="Camera Feed" />
            </div>
            <div class="side-panel">
                <div class="controls">
                    <button id="classifyBtn" onclick="classifyOnce()">🔍 Classify</button>
                    <button id="autoBtn" onclick="toggleAuto()">▶ Auto Detect</button>
                </div>
                <div class="status" id="status">Ready</div>
                <div class="result-card" id="resultCard">
                    <h2>🏷️ Detection Result</h2>
                    <div class="category-label" id="catLabel">—</div>
                    <div class="confidence" id="confLabel"></div>
                    <div class="bar-chart" id="barChart"></div>
                </div>
                <div class="history">
                    <h2>📋 History</h2>
                    <div id="historyList"></div>
                </div>
            </div>
        </div>
        <script>
            const COLORS = {{
                cardboard: '#e67e22', glass: '#3498db', metal: '#95a5a6',
                paper: '#f1c40f', plastic: '#e74c3c', trash: '#7f8c8d'
            }};
            const ICONS = {{
                cardboard: '📦', glass: '🥛', metal: '🥫',
                paper: '📄', plastic: '🧴', trash: '🗑️'
            }};

            let autoInterval = null;
            let classifying = false;

            async function doClassify() {{
                if (classifying) return;
                classifying = true;
                document.getElementById('status').textContent = '⏳ Classifying...';
                document.getElementById('classifyBtn').disabled = true;
                try {{
                    const resp = await fetch('/capture-and-classify', {{ method: 'POST' }});
                    const data = await resp.json();
                    if (resp.ok) {{
                        showResult(data);
                        addHistory(data);
                    }} else {{
                        document.getElementById('status').textContent = '❌ ' + data.detail;
                    }}
                }} catch (e) {{
                    document.getElementById('status').textContent = '❌ ' + e.message;
                }}
                classifying = false;
                document.getElementById('classifyBtn').disabled = false;
            }}

            function classifyOnce() {{ doClassify(); }}

            function toggleAuto() {{
                const btn = document.getElementById('autoBtn');
                if (autoInterval) {{
                    clearInterval(autoInterval);
                    autoInterval = null;
                    btn.textContent = '▶ Auto Detect';
                    btn.classList.remove('active');
                    document.getElementById('status').textContent = 'Auto detect stopped';
                }} else {{
                    doClassify();
                    autoInterval = setInterval(doClassify, 3000);
                    btn.textContent = '⏹ Stop';
                    btn.classList.add('active');
                }}
            }}

            function showResult(data) {{
                const cat = data.predicted_category;
                const pct = (data.confidence * 100).toFixed(1);
                const icon = ICONS[cat] || '❓';
                const color = COLORS[cat] || '#00d4aa';

                document.getElementById('catLabel').innerHTML =
                    `<span style="color:${{color}}">${{icon}} ${{cat.toUpperCase()}}</span>`;
                document.getElementById('confLabel').textContent = pct + '% confidence';
                document.getElementById('status').textContent =
                    `Detected: ${{cat}} (${{pct}}%)`;

                // Overlay on video
                const overlay = document.getElementById('overlay');
                overlay.innerHTML = `${{icon}} ${{cat.toUpperCase()}} ${{pct}}%`;
                overlay.style.color = color;
                overlay.classList.add('show');

                // Bar chart
                let bars = '';
                data.all_predictions.forEach(p => {{
                    const w = (p.confidence * 100).toFixed(1);
                    const c = COLORS[p.category] || '#00d4aa';
                    bars += `
                        <div class="bar-row">
                            <span class="bar-label">${{ICONS[p.category] || ''}} ${{p.category}}</span>
                            <div class="bar-track">
                                <div class="bar-fill" style="width:${{w}}%;background:${{c}}"></div>
                            </div>
                            <span class="bar-pct">${{w}}%</span>
                        </div>`;
                }});
                document.getElementById('barChart').innerHTML = bars;
            }}

            function addHistory(data) {{
                const list = document.getElementById('historyList');
                const cat = data.predicted_category;
                const pct = (data.confidence * 100).toFixed(1);
                const time = new Date().toLocaleTimeString();
                const icon = ICONS[cat] || '❓';
                const item = document.createElement('div');
                item.className = 'history-item';
                item.innerHTML = `<span>${{icon}} ${{cat}}</span><span>${{pct}}%</span><span>${{time}}</span>`;
                list.insertBefore(item, list.firstChild);
                if (list.children.length > 20) list.removeChild(list.lastChild);
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)