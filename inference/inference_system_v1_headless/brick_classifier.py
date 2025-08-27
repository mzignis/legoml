import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from picamera2 import Picamera2
import time
from PIL import Image
import numpy as np
import os
import cv2
from datetime import datetime
import threading

# ----------------------------
# 1. SimpleBrickNet Model Definition
# ----------------------------

class SimpleBrickNet(nn.Module):
    def __init__(self, num_classes=13):
        super(SimpleBrickNet, self).__init__()

        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 13)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------
# 2. BrickClassifier - Main Class and Initialization
# ----------------------------

class BrickClassifier:
    def __init__(self, model_path, snapshot_dir="snapshots"):
        self.model_path = model_path
        self.snapshot_dir = snapshot_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class labels
        self.class_labels = [
            'white_1x3_good', 'white_2x2_good', 'white_2x4_good',
            'blue_2x2_good', 'blue_2x6_good', 'blue_1x6_good',
            'white_1x3_damaged', 'white_2x2_damaged', 'white_2x4_damaged',
            'blue_2x2_damaged', 'blue_2x6_damaged', 'blue_1x6_damaged',
            'no_brick'
        ]
        
        # State variables
        self.model = None
        self.picam2 = None
        self.is_running = False
        self.capture_thread = None
        self.current_top4_classes = []
        self.current_top4_confidences = []
        self._lock = threading.Lock() # Thread safety
        
        # Initialize
        self._setup_snapshot_directory()
        self._load_model()
        self._setup_camera()
        

# ----------------------------
# 3. Directory and Model Setup Methods
# ----------------------------

    def _setup_snapshot_directory(self):
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
        

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
            else:
                model_state_dict = checkpoint
        else:
            model_state_dict = checkpoint
        
        self.model = SimpleBrickNet(num_classes=13)
        
        try:
            self.model.load_state_dict(model_state_dict, strict=True)
        except RuntimeError:
            self.model.load_state_dict(model_state_dict, strict=False)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        

# ----------------------------
# 4. Camera Setup and Configuration
# ----------------------------

    def _setup_camera(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (1024, 768)})
        self.picam2.configure(config)
        self.picam2.set_controls({
            "ScalerCrop": (0, 0, self.picam2.camera_properties['PixelArraySize'][0],
                          self.picam2.camera_properties['PixelArraySize'][1]),
            "Sharpness": 3,
            "ExposureTime": 15000,
            "AnalogueGain": 1.0,
            "Brightness": 0.25,
            "Contrast": 1.2
        })
        self.picam2.start()
        

# ----------------------------
# 5. Snapshot Management Methods
# ----------------------------

    def _cleanup_old_snapshots(self, max_files=6):
        try:
            snapshot_files = [f for f in os.listdir(self.snapshot_dir) if f.lower().endswith('.jpg')]
            
            if len(snapshot_files) > max_files:
                file_paths = [os.path.join(self.snapshot_dir, f) for f in snapshot_files]
                file_paths.sort(key=os.path.getmtime)
                
                files_to_delete = file_paths[:-max_files]
                for file_path in files_to_delete:
                    os.remove(file_path)
            
        except Exception as e:
            print(f"Error cleaning up old snapshots: {e}")
    
    def _save_cropped_snapshot(self, cropped_image, predicted_class, confidence):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            clean_class_name = predicted_class.replace(' ', '_').replace('/', '_')
            filename = f"{timestamp}_{clean_class_name}_conf{confidence:.3f}.jpg"
            filepath = os.path.join(self.snapshot_dir, filename)
            
            cropped_image.save(filepath, "JPEG", quality=95)
            self._cleanup_old_snapshots(max_files=6)
            
            return filepath
        
        except Exception as e:
            print(f"Error saving snapshot: {e}")
            return None
        

# ----------------------------
# 6. Image Processing and Prediction Methods
# ----------------------------

    def _crop_camera_image(self, pil_image):
        try:
            width, height = pil_image.size
            crop_params = {'left': 164, 'top': 164, 'right': 164, 'bottom': 164}
            
            crop_left = crop_params['left']
            crop_top = crop_params['top']
            crop_right = width - crop_params['right']
            crop_bottom = height - crop_params['bottom']
            
            if crop_left >= crop_right or crop_top >= crop_bottom:
                return pil_image
            
            crop_left = max(0, crop_left)
            crop_top = max(0, crop_top)
            crop_right = min(width, crop_right)
            crop_bottom = min(height, crop_bottom)
            
            return pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        except Exception:
            return pil_image
        

    def _preprocess_camera_image(self, image_array, target_size=(224, 224)):
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        pil_image = Image.fromarray(rgb_image)
        
        cropped_image = self._crop_camera_image(pil_image)
        
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(cropped_image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, cropped_image
        

    def _predict_image(self, image_array):
        img_tensor, cropped_pil_image = self._preprocess_camera_image(image_array)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)
            probabilities = F.softmax(predictions, dim=1)
            top4_confidences, top4_indices = torch.topk(probabilities, 4, dim=1)
            
            top4_confidences = top4_confidences[0].tolist()
            top4_indices = top4_indices[0].tolist()
            top4_classes = [self.class_labels[idx] for idx in top4_indices]
            
            # Thread-safe update of current results
            with self._lock:
                self.current_top4_classes = top4_classes
                self.current_top4_confidences = top4_confidences
            
            # Print results
            print("Top 4 predictions:")
            for i, (class_name, confidence) in enumerate(zip(top4_classes, top4_confidences), 1):
                print(f" {i}. {class_name} (confidence: {confidence:.3f})")
            
            # Save snapshot using top prediction
            saved_path = self._save_cropped_snapshot(cropped_pil_image, top4_classes[0], top4_confidences[0])
            if saved_path:
                print("Snapshot saved")
            
            return top4_classes, top4_confidences
        

# ----------------------------
# 7. Continuous Capture Control Methods
# ----------------------------

    def _continuous_capture_loop(self, prediction_interval):
        last_prediction_time = 0
        
        while self.is_running:
            try:
                image_array = self.picam2.capture_array()
                current_time = time.time()
                
                # Check if it's time for a new prediction
                if current_time - last_prediction_time >= prediction_interval:
                    self._predict_image(image_array)
                    last_prediction_time = current_time
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in continuous capture: {e}")
                break
        
        print("Continuous capture stopped")
        

    def start_continuous_capture(self, prediction_interval=2.0):
        if self.is_running:
            print("Continuous capture is already running")
            return
        
        self.is_running = True
        print(f"Starting continuous capture with {prediction_interval}s between predictions...")
        
        # Start capture in background thread
        self.capture_thread = threading.Thread(
            target=self._continuous_capture_loop,
            args=(prediction_interval,),
            daemon=True
        )
        self.capture_thread.start()
        print("Continuous capture started in background")
        

    def stop_continuous_capture(self):
        if self.is_running:
            self.is_running = False
            if self.capture_thread:
                self.capture_thread.join(timeout=5) # Wait up to 5 seconds
        

# ----------------------------
# 8. Public Interface Methods
# ----------------------------

    def get_latest_top4(self):
        with self._lock:
            return self.current_top4_classes.copy(), self.current_top4_confidences.copy()
        

    def is_capturing(self):
        return self.is_running
        

    def cleanup(self):
        self.stop_continuous_capture()
        if self.picam2:
            self.picam2.stop()