# Phase 7-8: UI, Deployment & Testing

## Phase 7: UI & Deployment (Week 7)

### 7.1 Minimal Web UI
**Owner:** Frontend Lead  
**Duration:** 4 hours  
**Dependencies:** Phase 6 complete

#### Tasks:
- [ ] 7.1.1 Create single-page HTML/JS UI
- [ ] 7.1.2 Add microphone access with permissions
- [ ] 7.1.3 Implement audio playback queue
- [ ] 7.1.4 Add visual feedback (waveform, status)
- [ ] 7.1.5 Make responsive for mobile

#### Implementation:
```html
<!-- src/izwi/ui/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Izwi - Shona Voice Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 100%;
            padding: 40px;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        
        .status {
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            transition: all 0.3s;
        }
        
        .status.connecting {
            background: #fff3cd;
            color: #856404;
        }
        
        .status.connected {
            background: #d4edda;
            color: #155724;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
        }
        
        .visualizer {
            height: 100px;
            background: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
        }
        
        .visualizer canvas {
            width: 100%;
            height: 100%;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .btn {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }
        
        .btn-record {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-record:hover {
            transform: scale(1.1);
        }
        
        .btn-record.recording {
            background: #dc3545;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); }
            70% { box-shadow: 0 0 0 20px rgba(220, 53, 69, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
        }
        
        .transcript {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            min-height: 100px;
            max-height: 200px;
            overflow-y: auto;
            margin-bottom: 20px;
        }
        
        .transcript-item {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        
        .transcript-item.user {
            background: #e3f2fd;
            text-align: right;
        }
        
        .transcript-item.assistant {
            background: #f3e5f5;
        }
        
        .settings {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #666;
        }
        
        @media (max-width: 600px) {
            .container {
                padding: 20px;
            }
            
            h1 {
                font-size: 24px;
            }
            
            .btn {
                width: 60px;
                height: 60px;
                font-size: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Izwi</h1>
        <p class="subtitle">Shona Voice Assistant</p>
        
        <div class="status" id="status">
            Connecting...
        </div>
        
        <div class="visualizer">
            <canvas id="visualizer"></canvas>
        </div>
        
        <div class="controls">
            <button class="btn btn-record" id="recordBtn">
                <span id="recordIcon">🎤</span>
            </button>
        </div>
        
        <div class="transcript" id="transcript">
            <div class="transcript-item" style="text-align: center; color: #999;">
                Press the microphone to start speaking
            </div>
        </div>
        
        <div class="settings">
            <span>Session: <span id="sessionId">-</span></span>
            <span>Latency: <span id="latency">-</span>ms</span>
        </div>
    </div>
    
    <script>
        class IzwiUI {
            constructor() {
                this.ws = null;
                this.audioContext = null;
                this.mediaStream = null;
                this.processor = null;
                this.isRecording = false;
                this.sessionId = null;
                
                // Audio playback queue
                this.audioQueue = [];
                this.isPlaying = false;
                
                // Visualizer
                this.canvas = document.getElementById('visualizer');
                this.canvasCtx = this.canvas.getContext('2d');
                this.analyser = null;
                
                // Metrics
                this.lastPingTime = 0;
                this.latency = 0;
                
                this.init();
            }
            
            async init() {
                // Set up WebSocket
                this.connect();
                
                // Set up UI events
                document.getElementById('recordBtn').addEventListener('click', () => {
                    this.toggleRecording();
                });
                
                // Set up visualizer
                this.setupVisualizer();
                
                // Ping for latency
                setInterval(() => this.ping(), 5000);
            }
            
            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const url = `${protocol}//${window.location.host}/ws/audio`;
                
                this.updateStatus('Connecting...', 'connecting');
                
                this.ws = new WebSocket(url);
                
                this.ws.onopen = () => {
                    this.updateStatus('Connected', 'connected');
                };
                
                this.ws.onmessage = async (event) => {
                    if (event.data instanceof Blob) {
                        // Audio data
                        this.queueAudio(event.data);
                    } else {
                        // Control message
                        const msg = JSON.parse(event.data);
                        this.handleMessage(msg);
                    }
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateStatus('Connection error', 'error');
                };
                
                this.ws.onclose = () => {
                    this.updateStatus('Disconnected', 'error');
                    setTimeout(() => this.connect(), 2000);
                };
            }
            
            async toggleRecording() {
                if (this.isRecording) {
                    this.stopRecording();
                } else {
                    await this.startRecording();
                }
            }
            
            async startRecording() {
                try {
                    // Get microphone permission
                    this.mediaStream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        }
                    });
                    
                    // Create audio context
                    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                        sampleRate: 16000
                    });
                    
                    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
                    
                    // Create processor
                    this.processor = this.audioContext.createScriptProcessor(512, 1, 1);
                    
                    this.processor.onaudioprocess = (e) => {
                        if (this.isRecording && this.ws.readyState === WebSocket.OPEN) {
                            const float32 = e.inputBuffer.getChannelData(0);
                            const int16 = new Int16Array(float32.length);
                            
                            // Convert to int16
                            for (let i = 0; i < float32.length; i++) {
                                const s = Math.max(-1, Math.min(1, float32[i]));
                                int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                            }
                            
                            // Send to server
                            this.ws.send(int16.buffer);
                        }
                    };
                    
                    // Connect audio graph
                    source.connect(this.processor);
                    this.processor.connect(this.audioContext.destination);
                    
                    // Set up analyser for visualization
                    this.analyser = this.audioContext.createAnalyser();
                    source.connect(this.analyser);
                    
                    // Update UI
                    this.isRecording = true;
                    document.getElementById('recordBtn').classList.add('recording');
                    document.getElementById('recordIcon').textContent = '⏹️';
                    
                    // Send start message
                    this.ws.send(JSON.stringify({
                        type: 'start',
                        codec: 'pcm16',
                        sample_rate: 16000
                    }));
                    
                    // Start visualization
                    this.visualize();
                    
                } catch (error) {
                    console.error('Failed to start recording:', error);
                    this.updateStatus('Microphone access denied', 'error');
                }
            }
            
            stopRecording() {
                this.isRecording = false;
                
                // Stop media stream
                if (this.mediaStream) {
                    this.mediaStream.getTracks().forEach(track => track.stop());
                    this.mediaStream = null;
                }
                
                // Disconnect audio nodes
                if (this.processor) {
                    this.processor.disconnect();
                    this.processor = null;
                }
                
                // Close audio context
                if (this.audioContext) {
                    this.audioContext.close();
                    this.audioContext = null;
                }
                
                // Update UI
                document.getElementById('recordBtn').classList.remove('recording');
                document.getElementById('recordIcon').textContent = '🎤';
                
                // Send end turn
                if (this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'end_turn' }));
                }
            }
            
            async queueAudio(blob) {
                // Add to queue
                this.audioQueue.push(blob);
                
                // Start playback if not playing
                if (!this.isPlaying) {
                    this.playNextAudio();
                }
            }
            
            async playNextAudio() {
                if (this.audioQueue.length === 0) {
                    this.isPlaying = false;
                    return;
                }
                
                this.isPlaying = true;
                const blob = this.audioQueue.shift();
                
                try {
                    // Create audio context if needed
                    if (!this.audioContext) {
                        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    }
                    
                    // Decode audio
                    const arrayBuffer = await blob.arrayBuffer();
                    const audioBuffer = await this.decodeAudio(arrayBuffer);
                    
                    // Play audio
                    const source = this.audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(this.audioContext.destination);
                    
                    source.onended = () => {
                        // Play next in queue
                        this.playNextAudio();
                    };
                    
                    source.start();
                    
                } catch (error) {
                    console.error('Failed to play audio:', error);
                    this.playNextAudio(); // Skip to next
                }
            }
            
            async decodeAudio(arrayBuffer) {
                // Assume PCM16 for now
                const int16 = new Int16Array(arrayBuffer);
                const float32 = new Float32Array(int16.length);
                
                // Convert to float32
                for (let i = 0; i < int16.length; i++) {
                    float32[i] = int16[i] / 32768;
                }
                
                // Create audio buffer
                const audioBuffer = this.audioContext.createBuffer(
                    1, float32.length, 16000
                );
                audioBuffer.getChannelData(0).set(float32);
                
                return audioBuffer;
            }
            
            handleMessage(msg) {
                switch (msg.type) {
                    case 'connected':
                        this.sessionId = msg.session_id;
                        document.getElementById('sessionId').textContent = msg.session_id;
                        break;
                    
                    case 'asr.partial':
                        this.addTranscript('user', msg.text);
                        break;
                    
                    case 'llm.partial':
                        // Could show typing indicator
                        break;
                    
                    case 'audio.chunk':
                        // Audio metadata
                        break;
                    
                    case 'pong':
                        this.latency = Date.now() - this.lastPingTime;
                        document.getElementById('latency').textContent = this.latency;
                        break;
                    
                    case 'error':
                        console.error('Server error:', msg.message);
                        this.updateStatus(`Error: ${msg.message}`, 'error');
                        break;
                }
            }
            
            addTranscript(role, text) {
                const transcript = document.getElementById('transcript');
                
                // Clear placeholder
                if (transcript.children[0].style.textAlign === 'center') {
                    transcript.innerHTML = '';
                }
                
                const item = document.createElement('div');
                item.className = `transcript-item ${role}`;
                item.textContent = text;
                
                transcript.appendChild(item);
                transcript.scrollTop = transcript.scrollHeight;
            }
            
            updateStatus(text, className) {
                const status = document.getElementById('status');
                status.textContent = text;
                status.className = `status ${className}`;
            }
            
            ping() {
                if (this.ws.readyState === WebSocket.OPEN) {
                    this.lastPingTime = Date.now();
                    this.ws.send(JSON.stringify({ type: 'ping' }));
                }
            }
            
            setupVisualizer() {
                // Set canvas size
                this.canvas.width = this.canvas.offsetWidth;
                this.canvas.height = this.canvas.offsetHeight;
                
                // Style
                this.canvasCtx.strokeStyle = '#667eea';
                this.canvasCtx.lineWidth = 2;
            }
            
            visualize() {
                if (!this.isRecording || !this.analyser) {
                    // Clear canvas
                    this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    return;
                }
                
                requestAnimationFrame(() => this.visualize());
                
                const bufferLength = this.analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                this.analyser.getByteTimeDomainData(dataArray);
                
                this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.canvasCtx.beginPath();
                
                const sliceWidth = this.canvas.width / bufferLength;
                let x = 0;
                
                for (let i = 0; i < bufferLength; i++) {
                    const v = dataArray[i] / 128.0;
                    const y = v * this.canvas.height / 2;
                    
                    if (i === 0) {
                        this.canvasCtx.moveTo(x, y);
                    } else {
                        this.canvasCtx.lineTo(x, y);
                    }
                    
                    x += sliceWidth;
                }
                
                this.canvasCtx.lineTo(this.canvas.width, this.canvas.height / 2);
                this.canvasCtx.stroke();
            }
        }
        
        // Initialize UI
        const ui = new IzwiUI();
    </script>
</body>
</html>
```

#### Testing:
```python
# tests/test_ui.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pytest
import time

@pytest.fixture
def driver():
    """Create Selenium WebDriver."""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # Allow microphone access
    prefs = {
        "profile.default_content_setting_values.media_stream_mic": 1
    }
    options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()

def test_ui_loads(driver):
    """Test UI loads successfully."""
    driver.get("http://localhost:8000/")
    
    # Check title
    assert "Izwi" in driver.title
    
    # Check main elements exist
    assert driver.find_element(By.ID, "recordBtn")
    assert driver.find_element(By.ID, "status")
    assert driver.find_element(By.ID, "transcript")
    assert driver.find_element(By.ID, "visualizer")

def test_websocket_connection(driver):
    """Test WebSocket connects."""
    driver.get("http://localhost:8000/")
    
    # Wait for connection
    wait = WebDriverWait(driver, 10)
    status = wait.until(
        EC.text_to_be_present_in_element(
            (By.ID, "status"),
            "Connected"
        )
    )
    
    assert status
    
    # Check session ID populated
    session_id = driver.find_element(By.ID, "sessionId").text
    assert session_id != "-"

def test_record_button_toggle(driver):
    """Test record button toggles."""
    driver.get("http://localhost:8000/")
    
    record_btn = driver.find_element(By.ID, "recordBtn")
    
    # Initial state
    assert "recording" not in record_btn.get_attribute("class")
    
    # Click to start
    record_btn.click()
    time.sleep(0.5)
    
    # Should be recording (if mic permission granted)
    # Note: In headless mode, mic access may fail

def test_responsive_design(driver):
    """Test responsive design."""
    driver.get("http://localhost:8000/")
    
    # Desktop size
    driver.set_window_size(1920, 1080)
    container = driver.find_element(By.CLASS_NAME, "container")
    assert container.size["width"] <= 500
    
    # Mobile size
    driver.set_window_size(375, 667)
    assert container.size["width"] < 375

def test_ui_performance():
    """Test UI bundle size."""
    import os
    
    ui_file = "src/izwi/ui/index.html"
    
    # Check file size
    size = os.path.getsize(ui_file)
    size_kb = size / 1024
    
    # Should be small
    assert size_kb < 100  # <100KB
    
    print(f"UI size: {size_kb:.1f}KB")
```

#### Acceptance Criteria:
- ✅ Single HTML/JS file < 100KB
- ✅ Works on mobile (responsive)
- ✅ Microphone permission handling
- ✅ Clear recording indicator
- ✅ Waveform visualization
- ✅ Audio playback queue
- ✅ WebSocket reconnection
- ✅ Latency display
- ✅ Transcript display
- ✅ Status indicators

---

### 7.2 Docker & Deployment
**Owner:** DevOps Lead  
**Duration:** 4 hours  
**Dependencies:** 7.1 complete

#### Tasks:
- [ ] 7.2.1 Create multi-stage Docker build
- [ ] 7.2.2 Deploy to HuggingFace Spaces
- [ ] 7.2.3 Set up Fly.io deployment
- [ ] 7.2.4 Configure monitoring
- [ ] 7.2.5 Set up CI/CD pipeline

#### Docker Implementation:
```dockerfile
# Dockerfile
# Multi-stage build for minimal image size

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.6.0

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-dev

# Stage 2: Model downloader
FROM python:3.11-slim as models

WORKDIR /models

# Download models
RUN pip install --no-cache-dir huggingface-hub

COPY scripts/download_models.py .
RUN python download_models.py

# Stage 3: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy models
COPY --from=models /models /app/models

# Copy application code
COPY src/ /app/src/
COPY .env.example /app/.env

# Create non-root user
RUN useradd -m -u 1000 izwi && chown -R izwi:izwi /app
USER izwi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/healthz')"

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "src.izwi.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### HuggingFace Spaces Deployment:
```yaml
# app.py (for HF Spaces)
import os
import sys
sys.path.insert(0, 'src')

from izwi.api.app import app

# HF Spaces configuration
title = "Izwi - Shona Voice Assistant"
description = "Real-time Shona speech-to-speech assistant"
article = "https://github.com/kinga-ai/izwi"

# Gradio interface (optional)
if os.getenv("SPACE_ID"):
    import gradio as gr
    
    def process_audio(audio):
        # Process with Izwi
        return audio
    
    iface = gr.Interface(
        fn=process_audio,
        inputs=gr.Audio(source="microphone", type="filepath"),
        outputs=gr.Audio(type="filepath"),
        title=title,
        description=description,
        article=article
    )
    
    iface.launch()
else:
    # Run FastAPI directly
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

```yaml
# requirements.txt (for HF Spaces)
fastapi==0.104.0
uvicorn[standard]==0.24.0
pydantic==2.5.0
faster-whisper==0.10.0
llama-cpp-python==0.2.0
TTS==0.20.0
gradio==4.0.0
```

#### Fly.io Deployment:
```toml
# fly.toml
app = "izwi"
primary_region = "jnb"  # Johannesburg for Africa

[build]
  dockerfile = "Dockerfile"

[env]
  IZWI_DEBUG = "false"
  IZWI_CORS_ALLOWED_ORIGINS = "https://izwi.fly.dev"

[experimental]
  auto_rollback = true

[[services]]
  http_checks = []
  internal_port = 8000
  protocol = "tcp"
  script_checks = []

  [services.concurrency]
    hard_limit = 100
    soft_limit = 50
    type = "connections"

  [[services.ports]]
    force_https = true
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443

  [[services.tcp_checks]]
    grace_period = "5s"
    interval = "15s"
    restart_limit = 0
    timeout = "2s"

  [[services.http_checks]]
    interval = "30s"
    grace_period = "5s"
    method = "get"
    path = "/healthz"
    protocol = "http"
    timeout = "2s"
    tls_skip_verify = false

[metrics]
  port = 9091
  path = "/metrics"
```

#### CI/CD Pipeline:
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
    
    - name: Install dependencies
      run: poetry install
    
    - name: Run tests
      run: poetry run pytest tests/ -v
    
    - name: Check code quality
      run: |
        poetry run ruff check src/
        poetry run black --check src/
        poetry run mypy src/

  deploy-hf:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Push to HuggingFace
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        git remote add hf https://huggingface.co/spaces/izwi/demo
        git push hf main --force

  deploy-fly:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - uses: superfly/flyctl-actions/setup-flyctl@master
    
    - name: Deploy to Fly.io
      run: flyctl deploy --remote-only
      env:
        FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}

  docker:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to DockerHub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          izwi/izwi:latest
          izwi/izwi:${{ github.sha }}
        cache-from: type=registry,ref=izwi/izwi:buildcache
        cache-to: type=registry,ref=izwi/izwi:buildcache,mode=max
```

#### Monitoring Setup:
```python
# scripts/setup_monitoring.py
"""Set up monitoring dashboards."""

import json

# Prometheus queries
queries = {
    "request_rate": "rate(izwi_requests_total[5m])",
    "error_rate": "rate(izwi_requests_total{status=~'5..'}[5m])",
    "p95_latency": "histogram_quantile(0.95, izwi_request_duration_seconds)",
    "active_sessions": "izwi_active_sessions",
    "asr_rtf": "izwi_asr_rtf",
    "llm_tokens_per_sec": "izwi_llm_tokens_per_sec",
    "memory_usage": "process_resident_memory_bytes",
    "cpu_usage": "rate(process_cpu_seconds_total[5m])"
}

# Grafana dashboard
dashboard = {
    "dashboard": {
        "title": "Izwi Metrics",
        "panels": [
            {
                "title": "Request Rate",
                "targets": [{"expr": queries["request_rate"]}]
            },
            {
                "title": "P95 Latency",
                "targets": [{"expr": queries["p95_latency"]}]
            },
            {
                "title": "Active Sessions",
                "targets": [{"expr": queries["active_sessions"]}]
            },
            {
                "title": "ASR RTF",
                "targets": [{"expr": queries["asr_rtf"]}]
            }
        ]
    }
}

print(json.dumps(dashboard, indent=2))
```

#### Testing:
```bash
# Test Docker build
docker build -t izwi:test .
docker run --rm -p 8000:8000 izwi:test

# Test container size
docker images izwi:test --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
# Should be < 500MB

# Test deployment
fly deploy --local-only
fly status
fly logs

# Load test deployed version
ab -n 1000 -c 50 https://izwi.fly.dev/healthz
```

#### Acceptance Criteria:
- ✅ Docker image < 500MB
- ✅ Multi-stage build working
- ✅ Starts in < 2s
- ✅ HF Spaces deployment works
- ✅ Fly.io free tier sufficient
- ✅ CI/CD runs < 5 min
- ✅ Auto-deploy on main branch
- ✅ Health checks passing
- ✅ Monitoring dashboards live
- ✅ Rollback capability

---

## Phase 8: Testing & Optimization (Week 8)

### 8.1 Performance Testing
**Owner:** QA Lead  
**Duration:** 6 hours  
**Dependencies:** Phase 7 complete

#### Tasks:
- [ ] 8.1.1 Load testing (100+ concurrent users)
- [ ] 8.1.2 Latency benchmarks (TTFW, E2E)
- [ ] 8.1.3 Memory profiling
- [ ] 8.1.4 CPU optimization
- [ ] 8.1.5 Network optimization

#### Load Testing:
```python
# tests/load_test.py
from locust import HttpUser, task, between, events
import time
import websocket
import json
import random

class VoiceUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session."""
        self.ws = None
        self.session_id = None
        self.connect_websocket()
    
    def connect_websocket(self):
        """Connect WebSocket."""
        ws_url = self.host.replace("http", "ws") + "/ws/audio"
        self.ws = websocket.create_connection(ws_url)
        
        # Get session ID
        msg = json.loads(self.ws.recv())
        self.session_id = msg.get("session_id")
    
    @task(1)
    def health_check(self):
        """Test health endpoint."""
        with self.client.get("/healthz", catch_response=True) as response:
            if response.elapsed.total_seconds() > 0.1:
                response.failure("Health check too slow")
    
    @task(10)
    def voice_interaction(self):
        """Test voice interaction."""
        start_time = time.time()
        
        # Send audio frames
        for _ in range(30):  # ~1 second of audio
            audio_frame = b"\x00" * 960  # 30ms silence
            self.ws.send_binary(audio_frame)
            time.sleep(0.03)
        
        # Wait for response
        response_received = False
        timeout = time.time() + 5  # 5 second timeout
        
        while time.time() < timeout:
            try:
                msg = self.ws.recv()
                if isinstance(msg, bytes):
                    response_received = True
                    break
            except:
                break
        
        # Record metrics
        total_time = time.time() - start_time
        
        if response_received:
            events.request.fire(
                request_type="WebSocket",
                name="voice_interaction",
                response_time=total_time * 1000,
                response_length=0,
                exception=None,
                context={}
            )
        else:
            events.request.fire(
                request_type="WebSocket",
                name="voice_interaction",
                response_time=total_time * 1000,
                response_length=0,
                exception="No response",
                context={}
            )
    
    def on_stop(self):
        """Clean up."""
        if self.ws:
            self.ws.close()

# Custom metrics
@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize custom metrics."""
    
    @events.request.add_listener
    def on_request(request_type, name, response_time, **kwargs):
        # Track P95 latency
        if name == "voice_interaction":
            if response_time < 1200:  # Under target
                environment.stats.log_request(
                    "custom", "latency_sla_met", response_time, 1
                )
            else:
                environment.stats.log_request(
                    "custom", "latency_sla_missed", response_time, 0
                )
```

#### Performance Benchmarks:
```python
# scripts/benchmark_e2e.py
import asyncio
import time
import numpy as np
from izwi.api.app import app
from fastapi.testclient import TestClient

async def benchmark_latency():
    """Benchmark end-to-end latency."""
    
    print("=" * 50)
    print("Izwi E2E Latency Benchmark")
    print("=" * 50)
    
    # Test scenarios
    scenarios = [
        ("Short greeting", "Mhoro"),
        ("Medium question", "How is the weather today in Harare?"),
        ("Long statement", "I would like to learn more about the history and culture of Zimbabwe, especially the Great Zimbabwe ruins."),
    ]
    
    results = []
    
    for name, text in scenarios:
        print(f"\nScenario: {name}")
        print(f"Input: {text}")
        
        # Generate audio from text (simulate)
        audio_duration = len(text.split()) * 0.3  # ~300ms per word
        
        # Measure latency
        ttfw_times = []
        e2e_times = []
        
        for i in range(10):  # 10 runs
            start = time.perf_counter()
            
            # Simulate pipeline
            # ASR
            await asyncio.sleep(0.2)  # ASR latency
            
            # LLM TTFB
            await asyncio.sleep(0.5)  # LLM TTFB
            ttfw = time.perf_counter() - start
            ttfw_times.append(ttfw * 1000)
            
            # LLM generation
            await asyncio.sleep(0.3)  # Token generation
            
            # TTS
            await asyncio.sleep(0.2)  # TTS synthesis
            
            e2e = time.perf_counter() - start
            e2e_times.append(e2e * 1000)
        
        # Calculate statistics
        ttfw_p50 = np.percentile(ttfw_times, 50)
        ttfw_p95 = np.percentile(ttfw_times, 95)
        e2e_p50 = np.percentile(e2e_times, 50)
        e2e_p95 = np.percentile(e2e_times, 95)
        
        print(f"  TTFW P50: {ttfw_p50:.0f}ms")
        print(f"  TTFW P95: {ttfw_p95:.0f}ms")
        print(f"  E2E P50: {e2e_p50:.0f}ms")
        print(f"  E2E P95: {e2e_p95:.0f}ms")
        
        # Check SLA
        ttfw_pass = "✅" if ttfw_p50 <= 500 else "❌"
        e2e_pass = "✅" if e2e_p95 <= 1200 else "❌"
        
        print(f"  SLA: TTFW {ttfw_pass}, E2E {e2e_pass}")
        
        results.append({
            "scenario": name,
            "ttfw_p50": ttfw_p50,
            "ttfw_p95": ttfw_p95,
            "e2e_p50": e2e_p50,
            "e2e_p95": e2e_p95
        })
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_ttfw_p50 = [r["ttfw_p50"] for r in results]
    all_e2e_p95 = [r["e2e_p95"] for r in results]
    
    overall_ttfw = np.mean(all_ttfw_p50)
    overall_e2e = np.mean(all_e2e_p95)
    
    print(f"Overall TTFW P50: {overall_ttfw:.0f}ms")
    print(f"Overall E2E P95: {overall_e2e:.0f}ms")
    
    if overall_ttfw <= 500 and overall_e2e <= 1200:
        print("✅ All latency targets met!")
    else:
        print("❌ Latency targets not met")

if __name__ == "__main__":
    asyncio.run(benchmark_latency())
```

#### Memory Profiling:
```python
# scripts/profile_memory.py
import tracemalloc
import asyncio
import psutil
import gc
from memory_profiler import profile

@profile
async def test_memory_usage():
    """Profile memory usage."""
    
    # Start tracing
    tracemalloc.start()
    process = psutil.Process()
    
    print("Initial memory:", process.memory_info().rss / 1024 / 1024, "MB")
    
    # Import and initialize
    from izwi.asr.engine import create_asr_engine
    from izwi.llm.engine import create_llm_engine
    from izwi.tts.engine import create_tts_engine
    
    asr = create_asr_engine()
    llm = create_llm_engine()
    tts = create_tts_engine()
    
    print("After init:", process.memory_info().rss / 1024 / 1024, "MB")
    
    # Process multiple sessions
    for i in range(10):
        # Simulate session
        audio = b"\x00" * 16000 * 5  # 5 seconds
        
        # Process
        # ... pipeline simulation ...
        
        if i % 5 == 0:
            gc.collect()
            print(f"After {i} sessions:", process.memory_info().rss / 1024 / 1024, "MB")
    
    # Get top memory allocations
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("\nTop 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
    
    tracemalloc.stop()

if __name__ == "__main__":
    asyncio.run(test_memory_usage())
```

#### Acceptance Criteria:
- ✅ P50 TTFW ≤ 500ms
- ✅ P95 E2E ≤ 1200ms
- ✅ 100+ concurrent users supported
- ✅ Memory < 4GB under load
- ✅ CPU < 80% under load
- ✅ Zero memory leaks detected
- ✅ Network optimized (compression)
- ✅ All SLAs documented

---

### 8.2 Security Audit
**Owner:** Security Lead  
**Duration:** 4 hours  
**Dependencies:** 8.1 complete

#### Tasks:
- [ ] 8.2.1 Penetration testing
- [ ] 8.2.2 Dependency vulnerability scanning
- [ ] 8.2.3 Secret scanning
- [ ] 8.2.4 CORS validation
- [ ] 8.2.5 Rate limit testing

#### Security Testing:
```bash
#!/bin/bash
# scripts/security_audit.sh

echo "======================================"
echo "Izwi Security Audit"
echo "======================================"

# 1. Dependency scanning
echo -e "\n[1] Scanning dependencies..."
poetry export -f requirements.txt | safety check --stdin
pip-audit

# 2. Secret scanning
echo -e "\n[2] Scanning for secrets..."
trufflehog filesystem . --json | jq '.[] | select(.verified == true)'
gitleaks detect --source . --verbose

# 3. SAST scanning
echo -e "\n[3] Static analysis..."
bandit -r src/ -f json | jq '.results[] | {severity, issue_text}'
semgrep --config=auto src/

# 4. Docker scanning
echo -e "\n[4] Scanning Docker image..."
docker build -t izwi:scan .
trivy image izwi:scan
grype izwi:scan

# 5. CORS testing
echo -e "\n[5] Testing CORS..."
curl -X OPTIONS http://localhost:8000/api/v1/chat \
  -H "Origin: http://evil.com" \
  -H "Access-Control-Request-Method: POST" \
  -I

# 6. Rate limiting
echo -e "\n[6] Testing rate limits..."
for i in {1..200}; do
  curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/healthz &
done | sort | uniq -c

# 7. Authentication bypass attempts
echo -e "\n[7] Testing authentication..."
# Try without auth
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "test"}]}' \
  -w "\nStatus: %{http_code}\n"

# 8. Input validation
echo -e "\n[8] Testing input validation..."
# Oversized payload
dd if=/dev/zero bs=1M count=10 | curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  --data-binary @- \
  -w "\nStatus: %{http_code}\n"

echo -e "\n======================================"
echo "Audit complete. Review results above."
echo "======================================"
```

#### Penetration Test Cases:
```python
# tests/test_security.py
import pytest
import httpx
import asyncio

def test_no_stack_traces():
    """Ensure no stack traces in errors."""
    response = httpx.get("http://localhost:8000/nonexistent")
    
    assert response.status_code == 404
    assert "Traceback" not in response.text
    assert "File" not in response.text
    assert "line" not in response.text

def test_cors_strict():
    """Test CORS is strict."""
    # Valid origin
    response = httpx.options(
        "http://localhost:8000/healthz",
        headers={"Origin": "http://localhost:7860"}
    )
    assert "access-control-allow-origin" in response.headers
    
    # Invalid origin
    response = httpx.options(
        "http://localhost:8000/healthz",
        headers={"Origin": "http://evil.com"}
    )
    assert "access-control-allow-origin" not in response.headers

def test_rate_limiting():
    """Test rate limiting works."""
    # Make many requests
    responses = []
    for _ in range(150):
        r = httpx.get("http://localhost:8000/healthz")
        responses.append(r.status_code)
    
    # Should see 429s
    assert 429 in responses

def test_input_validation():
    """Test input validation."""
    # Oversized text
    large_text = "x" * 10000
    response = httpx.post(
        "http://localhost:8000/v1/tts",
        json={"text": large_text}
    )
    
    assert response.status_code == 400

def test_no_directory_traversal():
    """Test path traversal protection."""
    response = httpx.get("http://localhost:8000/../../../etc/passwd")
    assert response.status_code == 404

def test_sql_injection():
    """Test SQL injection protection."""
    response = httpx.post(
        "http://localhost:8000/v1/chat",
        json={"messages": [{"role": "user", "content": "'; DROP TABLE users; --"}]}
    )
    
    # Should handle safely
    assert response.status_code in [200, 400]

def test_xxe_protection():
    """Test XXE protection."""
    xxe_payload = """<?xml version="1.0"?>
    <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
    <foo>&xxe;</foo>"""
    
    response = httpx.post(
        "http://localhost:8000/v1/chat",
        content=xxe_payload,
        headers={"Content-Type": "application/xml"}
    )
    
    assert response.status_code == 400

def test_csrf_protection():
    """Test CSRF protection."""
    # POST without CSRF token
    response = httpx.post(
        "http://localhost:8000/v1/chat",
        json={"messages": []},
        headers={"Origin": "http://evil.com"}
    )
    
    assert response.status_code in [401, 403]

@pytest.mark.asyncio
async def test_dos_protection():
    """Test DoS protection."""
    # Try to exhaust resources
    async with httpx.AsyncClient() as client:
        tasks = []
        for _ in range(1000):
            tasks.append(
                client.get("http://localhost:8000/healthz")
            )
        
        # Should handle gracefully
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some should succeed
        successes = [r for r in responses if not isinstance(r, Exception)]
        assert len(successes) > 0

def test_auth_bypass():
    """Test authentication bypass attempts."""
    # Try common bypasses
    bypasses = [
        {"Authorization": "Bearer null"},
        {"Authorization": "Bearer undefined"},
        {"Authorization": "Bearer "},
        {"X-API-Key": "test"},
    ]
    
    for headers in bypasses:
        response = httpx.get(
            "http://localhost:8000/v1/models",
            headers=headers
        )
        assert response.status_code == 401
```

#### SBOM Generation:
```python
# scripts/generate_sbom.py
"""Generate Software Bill of Materials."""

import json
import subprocess

def generate_sbom():
    """Generate SBOM in CycloneDX format."""
    
    # Get dependencies
    result = subprocess.run(
        ["poetry", "export", "-f", "requirements.txt", "--without-hashes"],
        capture_output=True,
        text=True
    )
    
    dependencies = []
    for line in result.stdout.split("\n"):
        if "==" in line:
            name, version = line.split("==")
            dependencies.append({
                "name": name,
                "version": version,
                "type": "library"
            })
    
    # Create SBOM
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": "urn:uuid:izwi-sbom-001",
        "version": 1,
        "metadata": {
            "timestamp": "2024-01-01T00:00:00Z",
            "tools": [{"name": "izwi-sbom-generator", "version": "1.0"}],
            "component": {
                "name": "izwi",
                "version": "0.1.0",
                "type": "application"
            }
        },
        "components": dependencies
    }
    
    with open("sbom.json", "w") as f:
        json.dump(sbom, f, indent=2)
    
    print(f"SBOM generated with {len(dependencies)} components")

if __name__ == "__main__":
    generate_sbom()
```

#### Acceptance Criteria:
- ✅ No high/critical vulnerabilities
- ✅ No secrets in code
- ✅ CORS properly configured
- ✅ Rate limiting effective
- ✅ Input validation working
- ✅ No stack traces exposed
- ✅ Authentication enforced
- ✅ SBOM generated
- ✅ Docker image scanned
- ✅ Penetration test passed

---

## Final Checklist

### MVP Release Criteria:
- [ ] **Performance**
  - [ ] P50 TTFW ≤ 500ms
  - [ ] P95 E2E ≤ 1200ms
  - [ ] 100+ concurrent users
  - [ ] Memory < 4GB
  - [ ] CPU < 80%

- [ ] **Security**
  - [ ] No critical vulnerabilities
  - [ ] Authentication working
  - [ ] Rate limiting active
  - [ ] CORS configured
  - [ ] No secrets exposed

- [ ] **Quality**
  - [ ] All tests passing
  - [ ] Code coverage > 80%
  - [ ] Documentation complete
  - [ ] .cursorrules compliance

- [ ] **Deployment**
  - [ ] Docker image < 500MB
  - [ ] HF Spaces live
  - [ ] Fly.io deployed
  - [ ] CI/CD working
  - [ ] Monitoring active

### Sign-off:
- [ ] Tech Lead: _____________
- [ ] Product Owner: _____________
- [ ] Security Lead: _____________
- [ ] QA Lead: _____________

---

**🎉 Congratulations! Izwi MVP is ready for launch! 🎉**
