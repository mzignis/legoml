import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from picamera2 import Picamera2
import time
from PIL import Image
import numpy as np
import os

# ----------------------------
# 1. Model Architecture
# ----------------------------
class BrickNet(nn.Module):
    def __init__(self, num_classes=13):
        super(BrickNet, self).__init__()
       
        self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ----------------------------
# 2. Load the trained model
# ----------------------------
model_path = "/home/candfpi4b/lego_pdm/brick_classifier_new.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, num_classes=13):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            model_state_dict = checkpoint
    else:
        model_state_dict = checkpoint
       
    model = BrickNet(num_classes=num_classes)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model

# ----------------------------
# 3. Define class labels
# ----------------------------
class_labels = [
    'white_1x3_good', 'white_2x2_good', 'white_2x4_good',
    'blue_2x2_good', 'blue_2x6_good', 'blue_1x6_good',
    'white_1x3_damaged', 'white_2x2_damaged', 'white_2x4_damaged',
    'blue_2x2_damaged', 'blue_2x6_damaged', 'blue_1x6_damaged',
    'no_brick'
]

# ----------------------------
# 4. Camera setup
# ----------------------------
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1024, 768)}
    )
    picam2.configure(config)
    picam2.set_controls({
        "ScalerCrop": (0, 0, picam2.camera_properties['PixelArraySize'][0],
                      picam2.camera_properties['PixelArraySize'][1]),
        "Sharpness": 3,
        "ExposureTime": 15000,
        "AnalogueGain": 1.0,
        "Brightness": 0.25,
        "Contrast": 1.2
    })
    picam2.start()
    time.sleep(2)
    print("Camera ready!")
    return picam2

# ----------------------------
# 5. Preprocess camera image
# ----------------------------
def preprocess_camera_image(image_array, target_size=(224, 224)):
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_array)
   
    # Apply transforms (same as your original)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
   
    img_tensor = transform(pil_image)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
    return img_tensor

# ----------------------------
# 6. Predict class
# ----------------------------
def predict_image(model, image_array):
    img_tensor = preprocess_camera_image(image_array)
   
    with torch.no_grad():
        predictions = model(img_tensor)
        probabilities = F.softmax(predictions, dim=1)
        confidence, class_idx = torch.max(probabilities, 1)
       
    class_idx = class_idx.item()
    confidence = confidence.item()
   
    predicted_class = class_labels[class_idx]
    print(f"Predicted class: {predicted_class} (confidence: {confidence:.2f})")
    return predicted_class, confidence

# ----------------------------
# 7. Single capture mode
# ----------------------------
def single_capture(model, picam2):
    print("Taking single photo...")
    start_time = time.time()
   
    image_array = picam2.capture_array()
    predicted_class, confidence = predict_image(model, image_array)
   
    processing_time = time.time() - start_time
    print(f"Processing time: {processing_time:.3f}s")

# ----------------------------
# 8. Continuous monitoring mode
# ----------------------------
def continuous_monitoring(model, picam2, interval=1.0):
    print("Starting continuous monitoring (Ctrl+C to stop)")
   
    try:
        frame_count = 0
        while True:
            start_time = time.time()
           
            image_array = picam2.capture_array()
            predicted_class, confidence = predict_image(model, image_array)
           
            processing_time = time.time() - start_time
            frame_count += 1
            print(f"Frame {frame_count} | Time: {processing_time:.3f}s")
           
            time.sleep(max(0, interval - processing_time))
           
    except KeyboardInterrupt:
        print("\nStopped monitoring.")

# ----------------------------
# 9. Main execution
# ----------------------------
def main():
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please update model_path variable")
        return
   
    # Load model
    model = load_model(model_path)
   
    # Setup camera
    picam2 = setup_camera()
   
    try:
        print("\nSelect mode:")
        print("1. Single capture")
        print("2. Continuous monitoring")
       
        choice = input("Enter choice (1 or 2): ").strip()
       
        if choice == "1":
            single_capture(model, picam2)
        elif choice == "2":
            interval = float(input("Interval in seconds (default 1.0): ") or "1.0")
            continuous_monitoring(model, picam2, interval)
        else:
            print("Invalid choice")
           
    except Exception as e:
        print(f"Error: {e}")
    finally:
        picam2.stop()
        picam2.close()
        print("Camera closed.")

if __name__ == "__main__":
    main()