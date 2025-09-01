import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import time
from PIL import Image
import numpy as np
import os

# ----------------------------
# 1. Model Architecture (Your actual SimpleBrickNet)
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

        # Based on the actual saved model structure:
        # classifier.1: Linear(4096, 256)
        # classifier.4: Linear(256, 13)
        self.classifier = nn.Sequential(
            nn.Flatten(),                             # layer 0
            nn.Linear(256 * 4 * 4, 256),            # layer 1: 4096 -> 256
            nn.ReLU(inplace=True),                   # layer 2
            nn.Dropout(0.3),                         # layer 3
            nn.Linear(256, num_classes)              # layer 4: 256 -> 13
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------------
# 2. Load the trained model
# ----------------------------
# Update this path to your Windows model location
model_path = r"C:\Users\atok\OneDrive - C&F S.A\Desktop\brick_classifier_simple96.pth"
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
    
    print("Loading SimpleBrickNet model...")
    model = SimpleBrickNet(num_classes=num_classes)
    
    try:
        model.load_state_dict(model_state_dict, strict=True)
        print("Model loaded successfully with strict matching!")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Trying with strict=False...")
        model.load_state_dict(model_state_dict, strict=False)
        print("Model loaded with non-strict mode")
    
    model = model.to(device)
    model.eval()
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
# 4. Camera setup for Windows with C920 (Enhanced)
# ----------------------------
def list_available_cameras():
    """List all available cameras using different backends"""
    available_cameras = []
    print("Scanning for available cameras...")
    
    # Try different backends - this is crucial for Windows!
    backends_to_try = [
        (cv2.CAP_DSHOW, "DirectShow"),        # Windows native - usually works best
        (cv2.CAP_MSMF, "Media Foundation"),   # Windows Media Foundation
        (cv2.CAP_ANY, "Default"),             # Default backend
    ]
    
    for backend_id, backend_name in backends_to_try:
        print(f"\nTrying {backend_name} backend...")
        found_any = False
        
        for i in range(10):  # Check first 10 indices
            try:
                cap = cv2.VideoCapture(i, backend_id)
                if cap.isOpened():
                    # Try to get camera properties
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Test if we can actually capture a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_info = {
                            'index': i,
                            'backend': backend_id,
                            'backend_name': backend_name,
                            'width': int(width) if width > 0 else 'Unknown',
                            'height': int(height) if height > 0 else 'Unknown',
                            'fps': int(fps) if fps > 0 else 'Unknown'
                        }
                        available_cameras.append(camera_info)
                        print(f"  Camera {i}: {camera_info['width']}x{camera_info['height']} @ {camera_info['fps']}fps")
                        found_any = True
                    cap.release()
            except Exception as e:
                # Silent fail and continue
                pass
        
        if found_any:
            break  # Found cameras with this backend, use it
    
    return available_cameras

def setup_camera():
    print("=== C920 Webcam Setup ===")
    print("Make sure:")
    print("1. Windows Camera app is completely closed")
    print("2. No other apps are using the camera") 
    print("3. Camera privacy settings allow desktop apps\n")
    
    available_cameras = list_available_cameras()
    
    if not available_cameras:
        print("\nNo cameras detected! Troubleshooting steps:")
        print("1. Unplug and reconnect your C920")
        print("2. Close Windows Camera app completely (check system tray)")
        print("3. Check Windows Settings > Privacy > Camera > Allow desktop apps")
        print("4. Try a different USB port")
        print("5. Restart the program")
        
        # Last resort: try manual backend specification
        print("\nTrying emergency detection...")
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Force DirectShow
                ret, frame = cap.read()
                if ret:
                    print(f"Found camera at index {i} using DirectShow!")
                    return cap
                cap.release()
            except:
                pass
        
        raise RuntimeError("Could not find any cameras. Please check troubleshooting steps above.")
    
    # Try to identify C920 by resolution capabilities
    c920_candidates = []
    
    for cam in available_cameras:
        # C920 typically supports high resolution
        if isinstance(cam['width'], int) and isinstance(cam['height'], int):
            if (cam['width'] >= 1280 and cam['height'] >= 720) or cam['width'] >= 1920:
                c920_candidates.append(cam)
    
    # Display available cameras
    print("\nAvailable cameras:")
    for i, cam in enumerate(available_cameras):
        is_candidate = cam in c920_candidates
        candidate_text = " <- Likely C920" if is_candidate else ""
        print(f"  [{cam['index']}] {cam['width']}x{cam['height']} @ {cam['fps']}fps ({cam['backend_name']}){candidate_text}")
    
    # Let user choose or auto-select
    if len(available_cameras) > 1:
        print(f"\nRecommended: Camera {c920_candidates[0]['index']} (likely C920)" if c920_candidates else "")
        choice = input("Enter camera index, or press Enter for auto-select: ").strip()
        
        if choice == "":
            selected_cam = c920_candidates[0] if c920_candidates else available_cameras[0]
        else:
            try:
                choice_idx = int(choice)
                selected_cam = next((cam for cam in available_cameras if cam['index'] == choice_idx), available_cameras[0])
            except (ValueError, IndexError):
                selected_cam = available_cameras[0]
    else:
        selected_cam = available_cameras[0]
    
    print(f"\nOpening camera {selected_cam['index']} using {selected_cam['backend_name']}...")
    
    # Open the selected camera with the correct backend
    cap = cv2.VideoCapture(selected_cam['index'], selected_cam['backend'])
    
    if not cap.isOpened():
        print(f"Failed to open camera {selected_cam['index']}, trying alternatives...")
        # Try other backends for this camera
        for backend_id, backend_name in [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation")]:
            cap = cv2.VideoCapture(selected_cam['index'], backend_id)
            if cap.isOpened():
                print(f"Success with {backend_name} backend!")
                break
            cap.release()
        else:
            raise RuntimeError(f"Failed to open camera {selected_cam['index']} with any backend")
    
    # Configure camera settings
    print("Configuring camera settings...")
    
    # Set resolution (try multiple options for C920)
    resolutions_to_try = [
        (1920, 1080),  # Full HD (C920 native)
        (1280, 720),   # HD 
        (1024, 768),   # Similar to original
        (800, 600),    # Fallback
        (640, 480)     # Last resort
    ]
    
    for width, height in resolutions_to_try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Test if setting worked
        if abs(actual_width - width) < 50 and abs(actual_height - height) < 50:  # Allow some tolerance
            print(f"Resolution set to: {actual_width}x{actual_height}")
            break
    else:
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Using camera's default resolution: {actual_width}x{actual_height}")
    
    # Set other properties
    settings = [
        (cv2.CAP_PROP_FPS, 30),
        (cv2.CAP_PROP_BUFFERSIZE, 1),      # Reduce buffer to get latest frames
        (cv2.CAP_PROP_AUTO_EXPOSURE, 1),   # Manual exposure
    ]
    
    for prop, value in settings:
        cap.set(prop, value)
    
    # Test capture
    print("Testing camera...")
    for i in range(5):  # Try a few frames to let camera settle
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print(f"✓ Camera working! Frame size: {test_frame.shape}")
            time.sleep(0.1)
        else:
            print(f"✗ Frame {i+1} failed")
    
    if not ret:
        raise RuntimeError("Camera opened but cannot capture frames!")
    
    print("Camera setup complete!\n")
    return cap

# ----------------------------
# 5. Preprocess camera image
# ----------------------------
def preprocess_camera_image(image_array, target_size=(224, 224)):
    # Convert BGR (OpenCV default) to RGB
    rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(rgb_image)
   
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
# 7. Single capture mode with preview
# ----------------------------
def single_capture(model, cap):
    print("=== Single Capture Mode ===")
    print("A preview window will open showing your camera feed.")
    print("Position your LEGO brick in the frame, then:")
    print("  - Press SPACE to capture and classify")
    print("  - Press 'q' to quit without capturing")
    print("  - Press 's' to save the current frame")
    
    cv2.namedWindow('LEGO Classifier - Preview', cv2.WINDOW_AUTOSIZE)
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Resize frame for display (laptop-friendly size)
        display_frame = cv2.resize(frame, (800, 600))
        
        # Add instruction text (adjust font size for smaller window)
        cv2.putText(display_frame, "Position LEGO brick in frame", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "SPACE = Capture | Q = Quit | S = Save", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the resized frame
        cv2.imshow('LEGO Classifier - Preview', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar - capture and classify
            print("\nCapturing and classifying...")
            start_time = time.time()
            
            # Use the original full-resolution frame for classification
            predicted_class, confidence = predict_image(model, frame)
            
            processing_time = time.time() - start_time
            print(f"Processing time: {processing_time:.3f}s")
            
            # Show result on resized frame
            result_frame = cv2.resize(frame, (800, 600))
            cv2.putText(result_frame, f"Prediction: {predicted_class}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Confidence: {confidence:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, "Press any key to continue...", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('LEGO Classifier - Result', result_frame)
            cv2.waitKey(0)  # Wait for any key press
            cv2.destroyWindow('LEGO Classifier - Result')
            
            # Ask if user wants to save the image
            save_choice = input("Save this capture? (y/n): ").lower().strip()
            if save_choice == 'y':
                timestamp = int(time.time())
                filename = f"capture_{predicted_class}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)  # Save original full-res frame
                print(f"Image saved as {filename}")
            
            # Ask if user wants to capture another
            continue_choice = input("Capture another image? (y/n): ").lower().strip()
            if continue_choice != 'y':
                break
                
        elif key == ord('q'):  # Quit
            print("Quitting single capture mode...")
            break
            
        elif key == ord('s'):  # Save current frame
            timestamp = int(time.time())
            filename = f"preview_save_{timestamp}.jpg"
            cv2.imwrite(filename, frame)  # Save original full-res frame
            print(f"Current frame saved as {filename}")
    
    cv2.destroyAllWindows()

# ----------------------------
# 8. Continuous monitoring mode with enhanced display
# ----------------------------
def continuous_monitoring(model, cap, interval=1.0):
    print("=== Continuous Monitoring Mode ===")
    print("Live camera feed with real-time classification!")
    print("Controls:")
    print("  - 'q' to quit")
    print("  - 's' to save current frame")
    print("  - 'p' to pause/unpause predictions")
    print("  - '+' to increase interval, '-' to decrease")
   
    cv2.namedWindow('LEGO Classifier - Live Feed', cv2.WINDOW_AUTOSIZE)
    
    frame_count = 0
    paused = False
    last_prediction = ("Starting...", 0.0)
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            # Resize frame for display (laptop-friendly: 800x600)
            display_frame = cv2.resize(frame, (800, 600))
            height, width = display_frame.shape[:2]  # Use resized dimensions
            
            # Only predict if not paused and enough time has passed
            if not paused:
                predicted_class, confidence = predict_image(model, frame)  # Use original frame for prediction
                last_prediction = (predicted_class, confidence)
                frame_count += 1
            else:
                predicted_class, confidence = last_prediction
            
            # Create info overlay background (adjusted for smaller window)
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)  # Smaller overlay
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Add prediction text with smaller fonts
            status = "PAUSED" if paused else "LIVE"
            cv2.putText(display_frame, f"Status: {status}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Color code confidence: Green = high, Yellow = medium, Red = low
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
                
            cv2.putText(display_frame, f"Prediction: {predicted_class}", (10, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display_frame, f"Confidence: {confidence:.3f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add frame counter and timing info (adjusted positions for smaller window)
            processing_time = time.time() - start_time
            if not paused:
                cv2.putText(display_frame, f"Frame: {frame_count} | Time: {processing_time:.2f}s", 
                           (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add interval info
            cv2.putText(display_frame, f"Interval: {interval:.1f}s", (10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add controls help (adjusted for smaller window)
            cv2.putText(display_frame, "Q=Quit S=Save P=Pause +=Faster -=Slower", 
                       (width - 350, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            # Show the resized frame
            cv2.imshow('LEGO Classifier - Live Feed', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting continuous monitoring...")
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"live_capture_{predicted_class}_{confidence:.2f}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)  # Save original full-resolution frame
                print(f"\nFrame saved as {filename}")
            elif key == ord('p'):
                paused = not paused
                status_msg = "PAUSED" if paused else "RESUMED"
                print(f"\nPrediction {status_msg}")
            elif key == ord('+') or key == ord('='):
                interval = max(0.1, interval - 0.1)
                print(f"\nInterval decreased to {interval:.1f}s (faster)")
            elif key == ord('-') or key == ord('_'):
                interval = min(5.0, interval + 0.1)
                print(f"\nInterval increased to {interval:.1f}s (slower)")
           
            # Wait for next frame (but don't wait if paused)
            if not paused:
                sleep_time = max(0, interval - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
           
    except KeyboardInterrupt:
        print("\nStopped monitoring (Ctrl+C pressed)")
    finally:
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames total")

# ----------------------------
# 9. Main execution
# ----------------------------
def main():
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please update the model_path variable at the top of the script")
        print("Current path:", model_path)
        return
   
    try:
        # Load model
        print("Loading model...")
        model = load_model(model_path)
       
        # Setup camera
        print("Setting up camera...")
        cap = setup_camera()
       
        print("\nSelect mode:")
        print("1. Single capture")
        print("2. Continuous monitoring")
       
        choice = input("Enter choice (1 or 2): ").strip()
       
        if choice == "1":
            single_capture(model, cap)
        elif choice == "2":
            interval = float(input("Interval in seconds (default 1.0): ") or "1.0")
            continuous_monitoring(model, cap, interval)
        else:
            print("Invalid choice")
           
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")

if __name__ == "__main__":
    main()