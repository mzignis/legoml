import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from picamera2 import Picamera2
import time
from PIL import Image
import os
import cv2
import threading

class BrickNet(nn.Module):
    """CNN optimized for LEGO brick recognition - exact copy from training script"""

    def __init__(self, num_classes=13, use_pretrained=True):
        super(BrickNet, self).__init__()

        if use_pretrained:
            # Use pretrained MobileNetV2 for better accuracy
            self.backbone = models.mobilenet_v2(pretrained=True)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.backbone.last_channel, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        else:
            # Custom lightweight architecture
            self.features = nn.Sequential(
                # First block
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                # Second block
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                # Third block
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                # Fourth block
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

            self.backbone = None  # Custom architecture

    def forward(self, x):
        if self.backbone is not None:
            return self.backbone(x)
        else:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x


def get_manual_transforms(image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform


class LegoBrickDefectInference:
    def __init__(self, model_path, device='cpu', num_classes=13):
        """
        Args:
            model_path (str): Path to the trained PyTorch model (.pth file)
            device (str): Device to run inference on ('cpu' or 'cuda')
            num_classes (int): Number of classes (default: 13)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Camera preview variables
        self.camera_running = False
        self.latest_frame = None
        self.latest_results = None
        self.frame_lock = threading.Lock()
        
        # Manual configuration (no metadata dependency)
        self.num_classes = num_classes
        self.input_size = 224  # Standard input size
        
        # Class names - EXACT MATCH from your training script
        self.class_names = [
            'white_1x3_good', 'white_2x2_good', 'white_2x4_good',
            'blue_2x2_good', 'blue_2x6_good', 'blue_1x6_good',
            'white_1x3_damaged', 'white_2x2_damaged', 'white_2x4_damaged',
            'blue_2x2_damaged', 'blue_2x6_damaged', 'blue_1x6_damaged',
            'no_brick'
        ]
        
        # Load the trained model (only state dict, no metadata)
        print(f"Loading model from {model_path}")
        if torch.cuda.is_available() and device == 'cuda':
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
            else:
                model_state_dict = checkpoint
        else:
            model_state_dict = checkpoint
            
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {len(self.class_names)} total")
        
        # Create the model with the same architecture as training
        self.model = BrickNet(num_classes=self.num_classes, use_pretrained=True)
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing pipeline - EXACTLY matching your training function
        self.transform = get_manual_transforms(image_size=self.input_size)
        
        # Create defect detection mapping
        self._create_defect_mapping()
        
        # Initialize camera
        self.setup_camera()
        
    def _create_defect_mapping(self):
        self.defect_classes = []
        self.good_classes = []
        self.no_brick_class = None
        
        for i, class_name in enumerate(self.class_names):
            if 'damaged' in class_name.lower():
                self.defect_classes.append(i)
            elif 'good' in class_name.lower():
                self.good_classes.append(i)
            elif 'no_brick' in class_name.lower():
                self.no_brick_class = i
        
        print(f"Defect classes: {[self.class_names[i] for i in self.defect_classes]}")
        print(f"Good classes: {[self.class_names[i] for i in self.good_classes]}")
        if self.no_brick_class is not None:
            print(f"No brick class: {self.class_names[self.no_brick_class]}")
        
    def setup_camera(self):
        print("Setting up camera...")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (1024, 768)}
        )
        self.picam2.configure(config)
        
        # Apply the SAME camera controls as data collection
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
        
        # Let camera warm up
        time.sleep(2)
        print("Camera ready with data collection settings!")
        
    def preprocess_image(self, image_array):
        """
        Preprocess the image using EXACT same transforms as training
        
        Args:
            image_array: RGB image array from camera
            
        Returns:
            tensor_image: Preprocessed tensor ready for model
        """
        # Convert numpy array to PIL Image (RGB format)
        pil_image = Image.fromarray(image_array)
        
        # Apply the exact same transforms as training
        # 1. Resize to (224, 224)
        # 2. Convert to tensor (scales to [0,1] and changes to CHW format)
        # 3. Normalize with ImageNet mean/std
        tensor_image = self.transform(pil_image)
        
        # Add batch dimension and move to device
        tensor_image = tensor_image.unsqueeze(0)
        return tensor_image.to(self.device)
    
    def predict(self, image_tensor):
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            predicted_class_name = self.class_names[predicted_class]
            
            # Determine defect status
            if predicted_class in self.defect_classes:
                defect_detected = True
                status = "DEFECT DETECTED"
            elif predicted_class in self.good_classes:
                defect_detected = False
                status = "NO DEFECT (GOOD BRICK)"
            elif predicted_class == self.no_brick_class:
                defect_detected = False
                status = "NO BRICK DETECTED"
            else:
                defect_detected = False
                status = "UNKNOWN CLASS"
            
            return {
                'defect_detected': defect_detected,
                'status': status,
                'predicted_class': predicted_class_name,
                'confidence': confidence,
                'class_probabilities': probabilities[0].cpu().numpy()  # All class probabilities
            }
    
    def capture_and_analyze(self):
        image_array = self.picam2.capture_array()
        
        # Update latest frame for camera preview
        with self.frame_lock:
            self.latest_frame = image_array.copy()
        
        # Preprocess and run inference
        image_tensor = self.preprocess_image(image_array)
        results = self.predict(image_tensor)
        
        # Update latest results for camera overlay
        with self.frame_lock:
            self.latest_results = results.copy()
        
        return results
    
    def start_camera_preview(self):
        self.camera_running = True
        self.camera_thread = threading.Thread(target=self._camera_preview_loop, daemon=True)
        self.camera_thread.start()
        print("Camera preview window started (press 'q' to close)")
        
    def stop_camera_preview(self):
        if hasattr(self, 'camera_running'):
            self.camera_running = False
            cv2.destroyAllWindows()
            
    def _camera_preview_loop(self):
        window_name = "LEGO Brick Defect Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_RESIZABLE)
        cv2.resizeWindow(window_name, 800, 600)
        
        while self.camera_running:
            try:
                # Get latest frame and results
                with self.frame_lock:
                    if self.latest_frame is not None:
                        frame = self.latest_frame.copy()
                        results = self.latest_results.copy() if self.latest_results else None
                    else:
                        frame = self.picam2.capture_array()
                        results = None
                
                # Use RGB frame directly (same as your data collection script)
                frame_display = frame
                
                # Add overlay with detection results
                if results:
                    self._add_overlay(frame_display, results)
                else:
                    cv2.putText(frame_display, "Initializing...", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow(window_name, frame_display)
                
                # Check for 'q' key press to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Camera preview closed by user")
                    self.camera_running = False
                    break
                    
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Error in camera preview: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        
    def _add_overlay(self, frame, results):
        height, width = frame.shape[:2]
        
        # Choose colors based on detection
        if results['defect_detected']:
            color = (0, 0, 255)  # Red for defect
            status_text = "DEFECT DETECTED"
        else:
            if 'NO BRICK' in results['status']:
                color = (0, 255, 255)  # Yellow for no brick
                status_text = "NO BRICK"
            else:
                color = (0, 255, 0)  # Green for good
                status_text = "GOOD BRICK"
        
        # Add semi-transparent overlay box
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        y_offset = 35
        cv2.putText(frame, status_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        y_offset += 25
        cv2.putText(frame, f"Class: {results['predicted_class']}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Confidence: {results['confidence']:.3f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add border around frame
        cv2.rectangle(frame, (0, 0), (width-1, height-1), color, 3)
    
    def run_continuous_inference(self, interval=1.0):
        """Run continuous defect detection with camera preview"""
        print("Starting continuous LEGO brick defect detection...")
        print("Press 'q' in camera window or Ctrl+C to stop")
        print("-" * 50)
        
        # Start camera preview
        self.start_camera_preview()
        time.sleep(1)  # Give camera preview time to start
        
        try:
            frame_count = 0
            while True:
                start_time = time.time()
                
                # Capture and analyze
                results = self.capture_and_analyze()
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Output results to console
                frame_count += 1
                print(f"Frame {frame_count:4d}: {results['status']}")
                print(f"             Class: {results['predicted_class']}")
                print(f"             Confidence: {results['confidence']:.3f}")
                print(f"             Processing: {processing_time:.3f}s")
                print("-" * 50)
                
                # Check if camera preview was closed
                if not self.camera_running:
                    print("Camera preview closed, stopping...")
                    break
                
                # Wait for next capture
                time.sleep(max(0, interval - processing_time))
                
        except KeyboardInterrupt:
            print("\nStopping defect detection...")
        finally:
            self.stop_camera_preview()
            self.cleanup()
    
    def run_single_inference(self):
        """Run inference on a single captured image"""
        print("Taking single photo for defect detection...")
        
        results = self.capture_and_analyze()
        
        # Output result
        print("\n" + "=" * 50)
        print("DETECTION RESULTS")
        print("=" * 50)
        
        if results['defect_detected']:
            print("DEFECT DETECTED!")
        else:
            if 'NO BRICK' in results['status']:
                print("NO BRICK IN VIEW")
            else:
                print("BRICK IS GOOD")
        
        print(f"Class: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.3f}")
        
        # Show top 3 predictions
        probs = results['class_probabilities']
        top3_indices = np.argsort(probs)[-3:][::-1]
        print("\nTop 3 predictions:")
        for i, idx in enumerate(top3_indices):
            print(f"  {i+1}. {self.class_names[idx]}: {probs[idx]:.3f}")
        
        print("=" * 50)
            
        return results
    
    def test_camera_colors(self):
        """Test camera color display to debug RGB/BGR issues"""
        print("Testing camera colors...")
        print("You should see correct colors in both windows")
        print("Press 'q' to close windows")
        
        try:
            while True:
                # Capture frame
                frame_rgb = self.picam2.capture_array()
                
                # Show original RGB (from PiCamera2 - same as data collection)
                cv2.imshow("RGB Direct (should look normal)", frame_rgb)
                
                # Show BGR converted (traditional OpenCV way)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("BGR converted (may look wrong)", frame_bgr)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
    
    def cleanup(self):
        print("Cleaning up...")
        self.stop_camera_preview()
        self.picam2.stop()
        self.picam2.close()
        print("Done.")


def main():
    print("=" * 60)
    print("LEGO BRICK DEFECT DETECTION - MANUAL PREPROCESSING")
    print("=" * 60)
    
    # Configuration - UPDATE THESE PATHS AND PARAMETERS
    MODEL_PATH = ""
    DEVICE = "cpu"  # Change to "cuda" if using GPU
    NUM_CLASSES = 13  # Matches your training data (13 classes total)
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please update MODEL_PATH in the script")
        return
    
    try:
        # Initialize detector with manual configuration
        detector = LegoBrickDefectInference(
            model_path=MODEL_PATH, 
            device=DEVICE,
            num_classes=NUM_CLASSES
        )
        
        # Simple menu
        print("\nSelect mode:")
        print("1. Single capture")
        print("2. Continuous monitoring")
        print("3. Test camera colors (debug)")
        
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            detector.run_single_inference()
            
        elif choice == "2":
            interval = float(input("Capture interval in seconds (default 1.0): ") or "1.0")
            detector.run_continuous_inference(interval)
            
        elif choice == "3":
            detector.test_camera_colors()
            
        else:
            print("Invalid choice")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Check model path and camera connection")
    
    finally:
        try:
            detector.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()