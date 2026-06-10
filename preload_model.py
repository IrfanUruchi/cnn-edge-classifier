from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

weights = MobileNet_V2_Weights.DEFAULT
mobilenet_v2(weights=weights)

print("MobileNetV2 weights cached successfully.")