# CNN Edge Classifier

CNN Edge Classifier is a lightweight computer vision inference workload for the Intelligent Fog Orchestration System.

It uses a pretrained MobileNetV2 convolutional neural network to classify uploaded images and returns the predicted class, confidence score, top-5 predictions, and inference time. The workload runs as a Dockerized edge service and includes both a drag-and-drop web UI and a REST prediction API.

This service is intentionally included as a CNN/DCNN workload in the fog orchestration platform. It shows that the same controller, MQTT communication, Docker runtime, and desired-state reconciliation model can manage computer vision inference workloads together with symbolic computation, Prolog reasoning, and scientific simulation services.

## Runtime Model

CNN Edge Classifier runs as an independent containerized workload.

The Fog Controller does not need special CNN-specific logic. It only includes `cnn-edge-classifier` in the desired state for a node. The IoT Smart Node receives that desired state through MQTT, pulls the Docker image, starts the container, exposes the service on port `8600`, and monitors the workload as part of the node runtime.

This keeps the workload portable. It can run locally, on a LAN machine, or on a remote fog node connected through Tailscale.

## Docker Image

```text
irfanuruchi/cnn-edge-classifier:latest
```

The image is published for:

```text
linux/amd64
linux/arm64
```

## Model

The workload uses:

```text
MobileNetV2
ImageNet pretrained weights
CPU inference
```

MobileNetV2 was selected because it is lightweight, portable, and suitable for edge inference scenarios. The model weights are preloaded into the Docker image so the container can start without downloading them at runtime.

## Run

```bash
docker run --rm \
  --name cnn-edge-classifier \
  -p 8600:8600 \
  irfanuruchi/cnn-edge-classifier:latest
```

Open the web UI:

```text
http://localhost:8600/ui
```

Health endpoint:

```text
http://localhost:8600/health
```

Root endpoint:

```text
http://localhost:8600/
```

## Web UI

The web UI supports drag-and-drop image upload. After an image is uploaded, the service runs CNN inference and displays the predicted class, confidence score, top-5 predictions, model name, task type, and inference time.

## Prediction API

Send an image to the `/predict` endpoint:

```bash
curl -X POST "http://localhost:8600/predict" \
  -F "file=@sample.jpg"
```

Example response:

```json
{
  "service": "cnn-edge-classifier",
  "model": "MobileNetV2",
  "task": "image-classification",
  "predicted_class": "German shepherd",
  "confidence": 0.3327,
  "top_5": [
    {
      "class": "German shepherd",
      "confidence": 0.3327
    },
    {
      "class": "malinois",
      "confidence": 0.1294
    },
    {
      "class": "Norwegian elkhound",
      "confidence": 0.0447
    }
  ],
  "inference_time_ms": 52.66
}
```

## Role in the Full System

In the full Intelligent Fog Orchestration System, this workload can be assigned to a fog node through the controller desired state.

```text
node-2:
  - integral-calculator
  - cnn-edge-classifier
```

The IoT Smart Node deploys the container, exposes the UI/API, and reports the workload status back to the Fog Controller. If the container stops unexpectedly and self-healing is enabled, the node can restart the managed workload.

## Endpoints

```text
GET  /
GET  /health
GET  /ui
POST /predict
```

## Port

```text
8600/tcp
```

## Related Repositories

```text
https://github.com/IrfanUruchi/intelligent-fog-orchestration-system
https://github.com/IrfanUruchi/fog-controller
https://github.com/IrfanUruchi/iot-smart-node
https://github.com/IrfanUruchi/fog-intelligence-llm
```

## Author

Irfan Uruçi  
South East European University  
Intelligent Systems Course Project  
Academic Year 2026
