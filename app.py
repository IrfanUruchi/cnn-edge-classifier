import io
import time
from typing import Any

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


app = FastAPI(
    title="CNN Edge Classifier",
    description="Lightweight CNN image classification workload for fog/edge orchestration.",
    version="1.0.0",
)

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.eval()

preprocess = weights.transforms()
categories = weights.meta["categories"]


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "cnn-edge-classifier",
        "model": "MobileNetV2",
        "weights": "ImageNet pretrained",
        "runtime": "CPU",
        "status": "running",
        "ui": "/ui",
        "predict_endpoint": "/predict",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "cnn-edge-classifier",
    }


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CNN Edge Classifier</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #0f172a;
      color: #e5e7eb;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .container {
      width: 92%;
      max-width: 850px;
      background: #111827;
      border: 1px solid #334155;
      border-radius: 18px;
      padding: 32px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }

    h1 {
      margin: 0;
      font-size: 30px;
      letter-spacing: -0.03em;
    }

    .subtitle {
      margin-top: 10px;
      color: #94a3b8;
      line-height: 1.5;
    }

    .drop-zone {
      margin-top: 28px;
      border: 2px dashed #475569;
      border-radius: 18px;
      padding: 42px;
      text-align: center;
      background: #020617;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .drop-zone.dragover {
      border-color: #38bdf8;
      background: #082f49;
    }

    .drop-title {
      font-size: 20px;
      font-weight: 700;
      margin-bottom: 8px;
    }

    .drop-hint {
      color: #94a3b8;
    }

    input[type="file"] {
      display: none;
    }

    .preview {
      margin-top: 24px;
      display: none;
      gap: 24px;
      align-items: flex-start;
    }

    .preview img {
      width: 260px;
      max-height: 260px;
      object-fit: contain;
      border-radius: 14px;
      border: 1px solid #334155;
      background: #020617;
    }

    .result {
      flex: 1;
      background: #020617;
      border: 1px solid #334155;
      border-radius: 14px;
      padding: 18px;
      min-height: 220px;
    }

    .status {
      margin-top: 18px;
      color: #94a3b8;
    }

    .main-prediction {
      font-size: 24px;
      font-weight: 800;
      margin-bottom: 6px;
    }

    .confidence {
      color: #38bdf8;
      font-weight: 700;
      margin-bottom: 18px;
    }

    .prediction-row {
      display: flex;
      justify-content: space-between;
      border-top: 1px solid #1e293b;
      padding: 10px 0;
      gap: 14px;
    }

    .prediction-row span:first-child {
      color: #e5e7eb;
    }

    .prediction-row span:last-child {
      color: #94a3b8;
      font-family: monospace;
    }

    .meta {
      margin-top: 14px;
      color: #94a3b8;
      font-size: 14px;
    }

    .button {
      margin-top: 18px;
      background: #2563eb;
      color: white;
      border: none;
      padding: 12px 18px;
      border-radius: 10px;
      font-weight: 700;
      cursor: pointer;
    }

    .button:hover {
      background: #1d4ed8;
    }

    @media (max-width: 760px) {
      .preview {
        flex-direction: column;
      }

      .preview img {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>CNN Edge Classifier</h1>
    <div class="subtitle">
      Lightweight MobileNetV2 image classification workload for fog and edge orchestration.
      Drag an image below to run local CNN inference.
    </div>

    <div id="dropZone" class="drop-zone">
      <div class="drop-title">Drop image here</div>
      <div class="drop-hint">or click to choose a JPG/PNG file</div>
      <input id="fileInput" type="file" accept="image/*" />
    </div>

    <div id="status" class="status">Waiting for image...</div>

    <div id="preview" class="preview">
      <img id="imagePreview" alt="Uploaded image preview" />
      <div class="result">
        <div id="resultContent">No prediction yet.</div>
      </div>
    </div>
  </div>

  <script>
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const statusBox = document.getElementById("status");
    const preview = document.getElementById("preview");
    const imagePreview = document.getElementById("imagePreview");
    const resultContent = document.getElementById("resultContent");

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropZone.classList.remove("dragover");

      const file = event.dataTransfer.files[0];
      if (file) {
        handleFile(file);
      }
    });

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (file) {
        handleFile(file);
      }
    });

    async function handleFile(file) {
      if (!file.type.startsWith("image/")) {
        statusBox.textContent = "Please upload an image file.";
        return;
      }

      preview.style.display = "flex";
      imagePreview.src = URL.createObjectURL(file);
      resultContent.innerHTML = "Running CNN inference...";
      statusBox.textContent = "Uploading image and running prediction...";

      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("/predict", {
          method: "POST",
          body: formData
        });

        if (!response.ok) {
          throw new Error("Prediction request failed.");
        }

        const data = await response.json();
        renderResult(data);
        statusBox.textContent = "Prediction complete.";
      } catch (error) {
        resultContent.innerHTML = "Error: " + error.message;
        statusBox.textContent = "Prediction failed.";
      }
    }

    function renderResult(data) {
      const top5 = data.top_5 || [];

      let html = `
        <div class="main-prediction">${data.predicted_class}</div>
        <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>
      `;

      if (top5.length > 0) {
        html += "<div>";
        top5.forEach(item => {
          html += `
            <div class="prediction-row">
              <span>${item.class}</span>
              <span>${(item.confidence * 100).toFixed(2)}%</span>
            </div>
          `;
        });
        html += "</div>";
      }

      html += `
        <div class="meta">
          Model: ${data.model}<br />
          Task: ${data.task || "image-classification"}<br />
          Inference time: ${data.inference_time_ms} ms
        </div>
      `;

      resultContent.innerHTML = html;
    }
  </script>
</body>
</html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    start_time = time.time()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    batch = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_probs, top_ids = torch.topk(probabilities, 5)
    inference_time_ms = round((time.time() - start_time) * 1000, 2)

    predictions = [
        {
            "class": categories[class_id.item()],
            "confidence": round(prob.item(), 4),
        }
        for prob, class_id in zip(top_probs, top_ids)
    ]

    return {
        "service": "cnn-edge-classifier",
        "model": "MobileNetV2",
        "task": "image-classification",
        "predicted_class": predictions[0]["class"],
        "confidence": predictions[0]["confidence"],
        "top_5": predictions,
        "inference_time_ms": inference_time_ms,
    }