# CNN Edge Classifier

CNN Edge Classifier is a lightweight computer vision inference workload for the Intelligent Fog Orchestration System.

It uses a pretrained MobileNetV2 convolutional neural network to classify input images and returns the predicted class, confidence score, and inference time.

The workload is designed to run as a Dockerized edge service and can be deployed by the IoT Smart Node using the same MQTT and desired-state reconciliation mechanism as the other workloads.

## Runtime

```bash
uvicorn app:app --host 0.0.0.0 --port 8600
