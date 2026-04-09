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
from app.models.classification import ClassificationResult, TrashCategory
from app.sensors.ultrasonic import GPIOUltrasonicSensor, SimulatedUltrasonicSensor
from app.services.actuator_service import actuators
from app.services.bin_level_service import bin_levels
from app.services.camera_service import camera
from app.services.classifier_service import classifier
from app.services.detection_service import detection
from app.services.led_service import leds
from app.services.monitor_service import MonitorService
from app.utils.logging import setup_logging

monitor = MonitorService()


def _create_sensor():
    """Create GPIO sensor on Raspberry Pi, fall back to simulated."""
    try:
        sensor = GPIOUltrasonicSensor(
            bin_id="SMART-BIN",
            echo_pin=settings.gpio_echo_pin,
            trig_pin=settings.gpio_trig_pin,
            capacity_cm=settings.bin_capacity_cm,
        )
        return sensor
    except Exception as exc:
        logger.warning(f"GPIO sensor unavailable ({exc}), using simulated sensor.")
        return SimulatedUltrasonicSensor(
            bin_id="SMART-BIN",
            capacity_cm=settings.bin_capacity_cm,
            initial_fill_percent=0.0,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    setup_logging()
    logger.info(f"Starting {settings.app_name} ...")

    # Create sensor (real GPIO on Pi, simulated otherwise)
    sensor = _create_sensor()
    monitor.register_sensor(sensor)
    monitor.start()

    # Load the trash classification model (non-blocking if missing)
    classifier.load_model()

    # Initialise camera (non-blocking if no camera attached)
    camera.initialise()

    # Initialise LEDs and bin level sensors
    leds.initialise()
    bin_levels.configure(leds)
    bin_levels.initialise()
    bin_levels.start()

    # Initialise actuators (with bin level checks)
    actuators.initialise()
    actuators.set_bin_level_service(bin_levels)

    # Configure and start auto-detection service (with actuators)
    detection.configure(sensor, camera, classifier, actuators)
    detection.start()

    yield
    detection.stop()
    bin_levels.stop()
    monitor.stop()
    camera.release()
    actuators.release()
    leds.release()
    bin_levels.release()
    if hasattr(sensor, "release"):
        sensor.release()
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
        "auto_detection": {
            "enabled": detection.is_enabled,
            "running": detection.is_running,
            "threshold_inches": settings.detection_threshold_inches,
            "object_detected": detection.object_detected,
            "distance_inches": detection.latest_distance_inches,
        },
        "endpoints": {
            "health": "/health",
            "all_bins": "/bins",
            "single_bin": "/bins/{bin_id}",
            "classify": "POST /classify",
            "capture_and_classify": "POST /capture-and-classify",
            "detection_status": "/detection/status",
            "detection_enable": "POST /detection/enable",
            "detection_disable": "POST /detection/disable",
            "detection_history": "/detection/history",
            "actuators_status": "/actuators/status",
            "actuator_trigger": "POST /actuators/trigger/{category}",
            "bin_levels": "/bins/levels",
            "leds_status": "/leds/status",
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



# ── Detection endpoints ──────────────────────────────────────────────────

@app.get("/detection/status", tags=["Detection"])
async def detection_status() -> dict[str, Any]:
    """Current state of the ultrasonic auto-detection system."""
    return {
        "enabled": detection.is_enabled,
        "running": detection.is_running,
        "object_detected": detection.object_detected,
        "distance_inches": detection.latest_distance_inches,
        "threshold_inches": settings.detection_threshold_inches,
        "latest_classification": (
            {
                "category": detection.latest_result.predicted_category.value,
                "confidence": detection.latest_result.confidence,
            }
            if detection.latest_result
            else None
        ),
    }


@app.post("/detection/enable", tags=["Detection"])
async def detection_enable() -> dict[str, str]:
    """Enable auto-detection."""
    detection.set_enabled(True)
    return {"status": "auto-detection enabled"}


@app.post("/detection/disable", tags=["Detection"])
async def detection_disable() -> dict[str, str]:
    """Disable auto-detection."""
    detection.set_enabled(False)
    return {"status": "auto-detection disabled"}


@app.get("/detection/history", tags=["Detection"])
async def detection_history() -> dict[str, Any]:
    """Return recent auto-detection results."""
    return {
        "count": len(detection.history),
        "detections": detection.history,
    }



# ── Actuator endpoints ───────────────────────────────────────────────────

@app.get("/actuators/status", tags=["Actuators"])
async def actuator_status() -> dict[str, Any]:
    """Return status of all 3 actuators."""
    return actuators.get_status()


@app.post("/actuators/trigger/{category}", tags=["Actuators"])
async def actuator_trigger(category: str) -> dict[str, Any]:
    """
    Manually trigger an actuator by trash category.

    Valid categories: biodegradable, non_biodegradable, hazardous.
    """
    try:
        cat = TrashCategory(category)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Use: biodegradable, non_biodegradable, hazardous.",
        )
    name = actuators.activate_for_category(cat)
    return {"category": category, "actuator": name, "status": "activated"}


# ── Bin Level endpoints ──────────────────────────────────────────────────

@app.get("/bins/levels", tags=["Bin Levels"])
async def get_bin_levels() -> dict[str, Any]:
    """Return fill level status of all 3 bins."""
    return bin_levels.get_status()


@app.get("/leds/status", tags=["LEDs"])
async def get_led_status() -> dict[str, Any]:
    """Return status of all 3 bin-full indicator LEDs."""
    return leds.get_status()




@app.post("/classify", response_model=ClassificationResult, tags=["Classification"])
async def classify_image(file: UploadFile = File(...)) -> ClassificationResult:
    """
    Upload an image of trash and get its predicted category.

    Categories: biodegradable, non_biodegradable, hazardous.
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
    Categories: biodegradable, non_biodegradable, hazardous.
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
                <div class="result-card" style="text-align:center;">
                    <h2>📡 Ultrasonic Sensor</h2>
                    <div style="font-size:2em;font-weight:bold;" id="distLabel">— in</div>
                    <div style="font-size:0.9em;color:#aaa;" id="sensorStatus">Waiting for sensor...</div>
                    <div style="margin-top:8px;">
                        Threshold: <strong>{settings.detection_threshold_inches} in</strong>
                    </div>
                </div>
                <div class="controls">
                    <button id="classifyBtn" onclick="classifyOnce()">🔍 Classify</button>
                    <button id="autoBtn" onclick="toggleAutoDetection()">📡 Auto: ON</button>
                </div>
                <div class="controls">
                    <button style="background:#27ae60" onclick="simulateCategory('biodegradable')">🍂 Biodegradable</button>
                    <button style="background:#2980b9" onclick="simulateCategory('non_biodegradable')">♻️ Non-Biodeg.</button>
                    <button style="background:#e74c3c" onclick="simulateCategory('hazardous')">☣️ Hazardous</button>
                </div>
                <div class="status" id="status">Ready – auto-detection active</div>
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
                biodegradable: '#27ae60',
                non_biodegradable: '#2980b9',
                hazardous: '#e74c3c'
            }};
            const ICONS = {{
                biodegradable: '🍂',
                non_biodegradable: '♻️',
                hazardous: '☣️'
            }};

            let classifying = false;
            let autoEnabled = true;
            let lastResultTs = null;

            // Poll detection status from the sensor service
            async function pollDetection() {{
                try {{
                    const resp = await fetch('/detection/status');
                    const d = await resp.json();
                    const dist = d.distance_inches;
                    const distEl = document.getElementById('distLabel');
                    const sensorEl = document.getElementById('sensorStatus');

                    if (dist !== null) {{
                        distEl.textContent = dist.toFixed(1) + ' in';
                        if (d.object_detected) {{
                            distEl.style.color = '#e74c3c';
                            sensorEl.textContent = '🔴 Object detected!';
                        }} else {{
                            distEl.style.color = '#00d4aa';
                            sensorEl.textContent = '🟢 Monitoring...';
                        }}
                    }} else {{
                        distEl.textContent = '— in';
                        sensorEl.textContent = 'Waiting for sensor...';
                    }}

                    // Check if there's a new auto-classification result
                    if (d.latest_classification) {{
                        const key = d.latest_classification.category + d.latest_classification.confidence;
                        if (key !== lastResultTs) {{
                            lastResultTs = key;
                            // Fetch full history to get the latest result details
                            const hResp = await fetch('/detection/history');
                            const hData = await hResp.json();
                            if (hData.detections.length > 0) {{
                                const latest = hData.detections[0];
                                showAutoResult(latest);
                                addHistory(latest);
                            }}
                        }}
                    }}
                }} catch (e) {{
                    // ignore polling errors
                }}
            }}
            setInterval(pollDetection, 500);

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
                        addHistory({{
                            category: data.predicted_category,
                            confidence: data.confidence,
                            distance_inches: null,
                            timestamp: new Date().toISOString(),
                        }});
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

            async function simulateCategory(category) {{
                document.getElementById('status').textContent = '⏳ Triggering ' + category + '...';
                try {{
                    const resp = await fetch('/actuators/trigger/' + category, {{ method: 'POST' }});
                    const data = await resp.json();
                    if (resp.ok) {{
                        const icon = ICONS[category] || '❓';
                        const color = COLORS[category] || '#00d4aa';
                        document.getElementById('catLabel').innerHTML =
                            `<span style="color:${{color}}">${{icon}} ${{category.toUpperCase()}}</span>`;
                        document.getElementById('confLabel').textContent = 'Simulated';
                        document.getElementById('status').textContent =
                            data.actuator === 'blocked'
                                ? '🚫 Bin full – relay blocked for ' + category
                                : '✅ Relay activated for ' + category;

                        const overlay = document.getElementById('overlay');
                        overlay.innerHTML = data.actuator === 'blocked'
                            ? `🚫 ${{category.toUpperCase()}} BIN FULL`
                            : `${{icon}} ${{category.toUpperCase()}} (simulated)`;
                        overlay.style.color = data.actuator === 'blocked' ? '#e74c3c' : color;
                        overlay.classList.add('show');
                        setTimeout(() => overlay.classList.remove('show'), 4000);

                        addHistory({{
                            category: category,
                            confidence: 1.0,
                            distance_inches: null,
                            timestamp: new Date().toISOString(),
                        }});
                    }} else {{
                        document.getElementById('status').textContent = '❌ ' + data.detail;
                    }}
                }} catch (e) {{
                    document.getElementById('status').textContent = '❌ ' + e.message;
                }}
            }}

            async function toggleAutoDetection() {{
                const btn = document.getElementById('autoBtn');
                autoEnabled = !autoEnabled;
                const action = autoEnabled ? 'enable' : 'disable';
                await fetch('/detection/' + action, {{ method: 'POST' }});
                btn.textContent = autoEnabled ? '📡 Auto: ON' : '📡 Auto: OFF';
                btn.classList.toggle('active', !autoEnabled);
                document.getElementById('status').textContent =
                    autoEnabled ? 'Auto-detection active' : 'Auto-detection paused';
            }}

            function showAutoResult(entry) {{
                const cat = entry.category;
                const pct = (entry.confidence * 100).toFixed(1);
                const icon = ICONS[cat] || '❓';
                const color = COLORS[cat] || '#00d4aa';
                const dist = entry.distance_inches ? entry.distance_inches.toFixed(1) + 'in' : '';

                document.getElementById('catLabel').innerHTML =
                    `<span style="color:${{color}}">${{icon}} ${{cat.toUpperCase()}}</span>`;
                document.getElementById('confLabel').textContent = pct + '% confidence' + (dist ? ' @ ' + dist : '');
                document.getElementById('status').textContent =
                    `Auto-detected: ${{cat}} (${{pct}}%)`;

                const overlay = document.getElementById('overlay');
                overlay.innerHTML = `${{icon}} ${{cat.toUpperCase()}} ${{pct}}%`;
                overlay.style.color = color;
                overlay.classList.add('show');
                setTimeout(() => overlay.classList.remove('show'), 4000);
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
                const cat = data.category || data.predicted_category;
                const pct = (data.confidence * 100).toFixed(1);
                const ts = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
                const icon = ICONS[cat] || '❓';
                const dist = data.distance_inches ? data.distance_inches.toFixed(1) + 'in' : '';
                const item = document.createElement('div');
                item.className = 'history-item';
                item.innerHTML = `<span>${{icon}} ${{cat}}</span><span>${{pct}}%</span><span>${{dist}}</span><span>${{ts}}</span>`;
                list.insertBefore(item, list.firstChild);
                if (list.children.length > 30) list.removeChild(list.lastChild);
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)