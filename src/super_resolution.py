import torch
from torchvision.transforms import ToTensor
import cv2

# Load Pre-trained Model
def load_esrgan_model():
    model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_esrgan")
    model.eval()
    return model

# Upscale Image
def upscale_image(image_path):
    model = load_esrgan_model()
    img = cv2.imread(image_path)
    img_tensor = ToTensor()(img).unsqueeze(0)  # Convert to tensor
    upscaled_img = model(img_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
    return upscaled_img
