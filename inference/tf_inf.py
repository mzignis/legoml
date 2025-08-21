import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from picamera2 import Picamera2
import time
from PIL import Image
import os

# ----------------------------
# 1. Load the trained model
# ----------------------------
model_path = "/home/candfpi4b/lego_pdm/trained_model_final.h5"

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
   
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model

# ----------------------------
# 2. Define class labels
# ----------------------------
class_labels = [
    "blue_1x6_damaged",
    "blue_1x6_good",
    "blue_2x2_damaged",
    "blue_2x2_good",
    "blue_2x6_damaged",
    "blue_2x6_good",
    "no_brick",
    "white_1x3_damaged",
    "white_1x3_good",
    "white_2x2_damaged",
    "white_2x2_good",
    "white_2x4_damaged",
    "white_2x4_good",
]

# ----------------------------
# 3. Camera setup
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
# 4. Preprocess camera image
# ----------------------------
def preprocess_camera_image(image_array, target_size=(224, 224)):
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_array)
   
    # Resize to target size
    pil_image = pil_image.resize(target_size)
   
    # Convert to array and preprocess
    img_array = image.img_to_array(pil_image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale to [0,1]
   
    return img_array

# ----------------------------
# 5. Load and preprocess file image (original function)
# ----------------------------
def load_and_preprocess(img_path, target_size=(224, 224)):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array /= 255.0  # rescale
    return img_array

# ----------------------------
# 6. Predict class
# ----------------------------
def predict_image_array(model, image_array):
    img_array = preprocess_camera_image(image_array)
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][class_idx]
   
    predicted_class = class_labels[class_idx]
    print(f"Predicted class: {predicted_class} (confidence: {confidence:.2f})")
    return predicted_class, confidence

def predict_image_file(model, img_path):
    img_array = load_and_preprocess(img_path)
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][class_idx]
   
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
    predicted_class, confidence = predict_image_array(model, image_array)
   
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
            predicted_class, confidence = predict_image_array(model, image_array)
           
            processing_time = time.time() - start_time
            frame_count += 1
            print(f"Frame {frame_count} | Time: {processing_time:.3f}s")
           
            time.sleep(max(0, interval - processing_time))
           
    except KeyboardInterrupt:
        print("\nStopped monitoring.")

# ----------------------------
# 9. File prediction mode (original functionality)
# ----------------------------
def predict_file(model, image_path):
    start_time = time.time()
    predicted_class, confidence = predict_image_file(model, image_path)
    processing_time = time.time() - start_time
    print(f"Processing time: {processing_time:.3f}s")

# ----------------------------
# 10. Main execution
# ----------------------------
def main():
    # Load model
    model = load_model(model_path)
   
    print("\nSelect mode:")
    print("1. Single camera capture")
    print("2. Continuous camera monitoring")
    print("3. Predict from file")
   
    choice = input("Enter choice (1, 2, or 3): ").strip()
   
    if choice == "1":
        picam2 = setup_camera()
        try:
            single_capture(model, picam2)
        finally:
            picam2.stop()
            picam2.close()
            print("Camera closed.")
           
    elif choice == "2":
        picam2 = setup_camera()
        try:
            interval = float(input("Interval in seconds (default 1.0): ") or "1.0")
            continuous_monitoring(model, picam2, interval)
        finally:
            picam2.stop()
            picam2.close()
            print("Camera closed.")
           
    elif choice == "3":
        image_path = input("Enter image path: ").strip()
        if not image_path:
            print("No image path provided")
            return
        try:
            predict_file(model, image_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
           
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()