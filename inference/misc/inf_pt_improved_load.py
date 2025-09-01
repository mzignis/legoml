import torch
import torch.nn.functional as F
from torchvision import transforms
from picamera2 import Picamera2
import time
from PIL import Image
import numpy as np
import os
import cv2
from datetime import datetime

# ----------------------------
# 1. Load the TorchScript model
# ----------------------------
# Update this path to your TorchScript model location
model_path = "/home/candfpi4b/lego_pdm/legoml/brick_classifier_simple_torchscript.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_torchscript_model(model_path):
    """Load a TorchScript model (.pt file)"""
    print(f"Loading TorchScript model from {model_path}...")
    
    try:
        # Load the TorchScript model
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print("TorchScript model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading TorchScript model: {e}")
        raise

# ----------------------------
# 2. Define class labels
# ----------------------------
class_labels = [
    'white_1x3_good', 'white_2x2_good', 'white_2x4_good',
    'blue_2x2_good', 'blue_2x6_good', 'blue_1x6_good',
    'white_1x3_damaged', 'white_2x2_damaged', 'white_2x4_damaged',
    'blue_2x2_damaged', 'blue_2x6_damaged', 'blue_1x6_damaged',
    'no_brick'
]

# ----------------------------
# 3. Snapshot saving setup
# ----------------------------
def setup_snapshot_directory(base_dir="snapshots"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def cleanup_old_snapshots(snapshot_dir="snapshots", max_files=6):
    try:
        snapshot_files = [f for f in os.listdir(snapshot_dir) if f.lower().endswith('.jpg')]
       
        if len(snapshot_files) > max_files:
            # Sort by modification time (oldest first)
            file_paths = [os.path.join(snapshot_dir, f) for f in snapshot_files]
            file_paths.sort(key=os.path.getmtime)
           
            # Delete oldest files to keep only max_files
            files_to_delete = file_paths[:-max_files]
            for file_path in files_to_delete:
                os.remove(file_path)
               
    except Exception as e:
        print(f"Error cleaning up old snapshots: {e}")

def save_cropped_snapshot(cropped_image, predicted_class, confidence, snapshot_dir="snapshots"):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits from microseconds
       
        clean_class_name = predicted_class.replace(' ', '_').replace('/', '_')
       
        filename = f"{timestamp}_{clean_class_name}_conf{confidence:.3f}.jpg"
        filepath = os.path.join(snapshot_dir, filename)
       
        cropped_image.save(filepath, "JPEG", quality=95)
       
        cleanup_old_snapshots(snapshot_dir, max_files=6)
       
        return filepath
       
    except Exception as e:
        print(f"Error saving snapshot: {e}")
        return None

# ----------------------------
# 4. Camera setup
# ----------------------------
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1024, 768)})
    picam2.configure(config)
    picam2.set_controls({
        "ScalerCrop": (0, 0, picam2.camera_properties['PixelArraySize'][0], picam2.camera_properties['PixelArraySize'][1]),
        "Sharpness": 3,
        "ExposureTime": 15000,
        "AnalogueGain": 1.0,
        "Brightness": 0.25,
        "Contrast": 1.2
    })
    picam2.start()
    return picam2

# ----------------------------
# 5. Preprocess camera image 
# ----------------------------
def preprocess_camera_image(image_array, target_size=(224, 224), return_cropped_pil=False):
    # Convert numpy array to PIL Image
    rgb_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    pil_image = Image.fromarray(rgb_image)
   
    crop_params = {
        'left': 164,  
        'top': 164,  
        'right': 164,  
        'bottom': 164  
    }
   
    cropped_image = crop_camera_image(pil_image, crop_params)
   
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
   
    img_tensor = transform(cropped_image)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
   
    if return_cropped_pil:
        return img_tensor, cropped_image
    return img_tensor

def crop_camera_image(pil_image, crop_params):
    """Crop PIL image by removing pixels from each side (same as training preprocessing)"""
    try:
        # Get original dimensions
        width, height = pil_image.size

        # Calculate crop coordinates by removing pixels from each side
        left = crop_params.get('left', 0) 
        top = crop_params.get('top', 0)  
        right_remove = crop_params.get('right', 0)
        bottom_remove = crop_params.get('bottom', 0)  

        crop_left = left
        crop_top = top
        crop_right = width - right_remove
        crop_bottom = height - bottom_remove

        # Validate crop coordinates
        if crop_left >= crop_right or crop_top >= crop_bottom:
            print(f"Warning: Invalid crop for camera image - using original")
            print(f"Original: {width}x{height}, Crop removes: L{left} T{top} R{right_remove} B{bottom_remove}")
            return pil_image

        # Ensure coordinates are within bounds
        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        crop_right = min(width, crop_right)
        crop_bottom = min(height, crop_bottom)

        cropped_img = pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))
       
        return cropped_img

    except Exception as e:
        print(f"Error cropping camera image: {e}")
        return pil_image  # Return original if cropping fails

# ----------------------------
# 6. Display image with prediction
# ----------------------------
def display_image_with_prediction(image_array, predicted_class, confidence, window_name="Brick Classifier", show_crop_region=False):
    display_image = image_array.copy()
   
    # Optionally show crop region overlay
    if show_crop_region:
        height, width = display_image.shape[:2]
        crop_params = {
            'left': 164, 'top': 164, 'right': 164, 'bottom': 164
        }
       
        # Draw crop region rectangle
        crop_left = crop_params['left']
        crop_top = crop_params['top']
        crop_right = width - crop_params['right']
        crop_bottom = height - crop_params['bottom']
       
        # Draw rectangle showing crop area
        cv2.rectangle(display_image, (crop_left, crop_top), (crop_right, crop_bottom), (255, 0, 0), 2)
        # Add crop region label
        cv2.putText(display_image, "Crop Region", (crop_left + 5, crop_top + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
   
    height, width = display_image.shape[:2]
    if width > 800:
        scale = 800 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        display_image = cv2.resize(display_image, (new_width, new_height))
   
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)  # Green if confident, yellow if not
    thickness = 2
   
    text1 = f"Class: {predicted_class}"
    text2 = f"Confidence: {confidence:.3f}"
   
    (text_width1, text_height1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
    (text_width2, text_height2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
   
    max_text_width = max(text_width1, text_width2)
    total_text_height = text_height1 + text_height2 + 20
   
    cv2.rectangle(display_image, (10, 10), (max_text_width + 20, total_text_height + 10), (0, 0, 0), -1)
   
    cv2.putText(display_image, text1, (15, 35), font, font_scale, color, thickness)
    cv2.putText(display_image, text2, (15, 65), font, font_scale, color, thickness)
   
    cv2.imshow(window_name, display_image)
   
    return display_image

# ----------------------------
# 7. Predict class with display
# ----------------------------
def predict_image(model, image_array, show_display=True, save_snapshot=True, snapshot_dir="snapshots"):
    img_tensor, cropped_pil_image = preprocess_camera_image(image_array, return_cropped_pil=True)
   
    with torch.no_grad():
        predictions = model(img_tensor)
        probabilities = F.softmax(predictions, dim=1)
       
        # Get top 4 predictions
        top4_confidences, top4_indices = torch.topk(probabilities, 4, dim=1)
       
    # Convert to lists
    top4_confidences = top4_confidences[0].tolist()  # Remove batch dimension
    top4_indices = top4_indices[0].tolist()
   
    # Get class names for top 4
    top4_classes = [class_labels[idx] for idx in top4_indices]
   
    # Print top 4 predictions
    print("Top 4 predictions:")
    for i, (class_name, confidence) in enumerate(zip(top4_classes, top4_confidences), 1):
        print(f"  {i}. {class_name} (confidence: {confidence:.3f})")
   
    # Use top prediction for saving and display
    predicted_class = top4_classes[0]
    confidence = top4_confidences[0]
   
    # Save the cropped snapshot if requested (using top prediction)
    if save_snapshot:
        saved_path = save_cropped_snapshot(cropped_pil_image, predicted_class, confidence, snapshot_dir)
        if saved_path:
            print("Snapshot saved")
   
    # Display the image with prediction if requested (showing top prediction)
    if show_display:
        display_image_with_prediction(image_array, predicted_class, confidence)
   
    return predicted_class, confidence, top4_classes, top4_confidences

# ----------------------------
# 8. Single capture mode 
# ----------------------------
def single_capture(model, picam2, snapshot_dir="snapshots"):
    print("Taking single photo...")
    print("Press any key to close the window after viewing the result")
    start_time = time.time()
   
    image_array = picam2.capture_array()
    predicted_class, confidence, top4_classes, top4_confidences = predict_image(model, image_array,
                                              show_display=True,
                                              save_snapshot=True,
                                              snapshot_dir=snapshot_dir)
   
    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
    return predicted_class, confidence

# ----------------------------
# 9. Continuous capture mode 
# ----------------------------
def continuous_capture(model, picam2, snapshot_dir="snapshots", prediction_interval=2.0):
    print(f"Starting continuous capture mode with {prediction_interval}s between predictions...")
    print("Press 's' to save current snapshot, 'q' to quit")
   
    last_prediction_time = 0
    current_prediction = None
    current_confidence = None
   
    while True:
        image_array = picam2.capture_array()
        current_time = time.time()
       
        # Check if it's time for a new prediction
        if current_time - last_prediction_time >= prediction_interval:
            # Make prediction and auto-save snapshot
            current_prediction, current_confidence, top4_classes, top4_confidences = predict_image(model, image_array,
                                                                 show_display=False,
                                                                 save_snapshot=True,
                                                                 snapshot_dir=snapshot_dir)
            last_prediction_time = current_time
       
        # Always display the image with the most recent prediction
        if current_prediction is not None:
            display_image_with_prediction(image_array, current_prediction, current_confidence)
        else:
            # Show image without prediction for first frame
            cv2.imshow("Brick Classifier", image_array)
       
        key = cv2.waitKey(1) & 0xFF
       
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s') and current_prediction is not None:
            # Manually save additional snapshot
            _, cropped_pil_image = preprocess_camera_image(image_array, return_cropped_pil=True)
            saved_path = save_cropped_snapshot(cropped_pil_image, current_prediction, current_confidence, snapshot_dir)
            if saved_path:
                print("Manual snapshot saved")
   
    cv2.destroyAllWindows()

# ----------------------------
# 10. Main function
# ----------------------------
def main():
    snapshot_dir = setup_snapshot_directory("snapshots")
   
    # Load the TorchScript model
    model = load_torchscript_model(model_path)
   
    picam2 = setup_camera()
   
    try:
        print("Choose mode:")
        print("1. Single capture (with auto-save)")
        print("2. Continuous capture with timing")
       
        choice = input("Enter choice (1 or 2): ").strip()
       
        if choice == "1":
            single_capture(model, picam2, snapshot_dir)
        elif choice == "2":
            # Get timing preference for continuous mode
            while True:
                try:
                    interval = float(input("Enter time between predictions (float sec): ").strip())
                    if interval > 0:
                        break
                    else:
                        print("Please enter a positive number")
                except ValueError:
                    print("Please enter a valid number")
           
            continuous_capture(model, picam2, snapshot_dir, prediction_interval=interval)
        else:
            print("Invalid choice")
           
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()