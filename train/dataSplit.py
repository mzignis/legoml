import os
import shutil
import random
import json
from PIL import Image

# ----------------------------
# LOAD CONFIGURATION
# ----------------------------
def load_config(config_path="split_config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found!")
        print("Using default configuration...")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        print("Using default configuration...")
        return get_default_config()

def get_default_config():
    """Return default configuration if JSON file is missing"""
    return {
        "dataset_config": {
            "raw_data_dir": "/brick_dataset/raw",
            "output_dir": "dataset_split",
            "random_seed": 42
        },
        "split_ratios": {
            "train_ratio": 0.6,
            "val_ratio": 0.3,
            "test_ratio": 0.1
        },
        "cropping_config": {
            "enable_cropping": True,
            "crop_params": {
                "left": 164,
                "top": 164,
                "right": 164,
                "bottom": 164
            }
        },
        "processing_options": {
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
            "debug_info": True,
            "show_first_crops": 5
        }
    }

# Load configuration
config = load_config()

# Extract configuration values
raw_data_dir = config["dataset_config"]["raw_data_dir"]
output_dir = config["dataset_config"]["output_dir"]
random_seed = config["dataset_config"]["random_seed"]

train_ratio = config["split_ratios"]["train_ratio"]
val_ratio = config["split_ratios"]["val_ratio"]
test_ratio = config["split_ratios"]["test_ratio"]

enable_cropping = config["cropping_config"]["enable_cropping"]
crop_params = config["cropping_config"]["crop_params"]

supported_formats = tuple(config["processing_options"]["supported_formats"])
debug_info = config["processing_options"]["debug_info"]
show_first_crops = config["processing_options"]["show_first_crops"]

# Validate ratios sum to 1.0
total_ratio = train_ratio + val_ratio + test_ratio
if abs(total_ratio - 1.0) > 0.001:
    print(f"WARNING: Split ratios don't sum to 1.0 (sum = {total_ratio})")
    print("Normalizing ratios...")
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

# ----------------------------
# DISPLAY CONFIGURATION
# ----------------------------
print("=" * 60)
print("DATASET SPLIT CONFIGURATION")
print("=" * 60)
print(f"Raw data directory: {raw_data_dir}")
print(f"Output directory: {output_dir}")
print(f"Random seed: {random_seed}")
print(f"Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
print(f"Cropping enabled: {enable_cropping}")
if enable_cropping:
    print(f"Crop parameters: L={crop_params['left']}, T={crop_params['top']}, "
          f"R={crop_params['right']}, B={crop_params['bottom']}")
print()

# ----------------------------
# CROPPING FUNCTION
# ----------------------------
def crop_image(image_path, output_path, crop_params):
    """Crop image by removing pixels from each side"""
    try:
        with Image.open(image_path) as img:
            # Get original dimensions
            width, height = img.size

            # Calculate crop coordinates by removing pixels from each side
            left = crop_params.get('left', 0)  # Remove from left
            top = crop_params.get('top', 0)  # Remove from top
            right_remove = crop_params.get('right', 0)  # Remove from right
            bottom_remove = crop_params.get('bottom', 0)  # Remove from bottom

            # Convert to PIL crop coordinates (left, top, right, bottom)
            # PIL crop uses absolute coordinates where right/bottom are the final positions
            crop_left = left
            crop_top = top
            crop_right = width - right_remove
            crop_bottom = height - bottom_remove

            # Validate crop coordinates
            if crop_left >= crop_right or crop_top >= crop_bottom:
                if debug_info:
                    print(f"Warning: Invalid crop for {image_path} - crop would result in zero/negative dimensions")
                    print(f"Original: {width}x{height}, Crop removes: L{left} T{top} R{right_remove} B{bottom_remove}")
                    print(f"Resulting size would be: {crop_right - crop_left}x{crop_bottom - crop_top}")
                # Copy original instead
                shutil.copy2(image_path, output_path)
                return False

            # Ensure coordinates are within bounds
            crop_left = max(0, crop_left)
            crop_top = max(0, crop_top)
            crop_right = min(width, crop_right)
            crop_bottom = min(height, crop_bottom)

            # Perform crop
            cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            cropped_img.save(output_path)

            # Debug info for first few images
            if debug_info and total_images < show_first_crops:
                print(f"  Cropped {os.path.basename(image_path)}: {width}x{height} → {crop_right - crop_left}x{crop_bottom - crop_top}")

            return True

    except Exception as e:
        if debug_info:
            print(f"Error cropping {image_path}: {e}")
        # Fallback: copy original image
        shutil.copy2(image_path, output_path)
        return False

# ----------------------------
# SETUP
# ----------------------------
random.seed(random_seed)
os.makedirs(output_dir, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# ----------------------------
# SPLIT AND PROCESS DATA
# ----------------------------
classes = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
print(f"Found {len(classes)} classes: {classes}")

total_images = 0
cropped_images = 0
failed_crops = 0

for cls in classes:
    cls_path = os.path.join(raw_data_dir, cls)

    # Get all image files
    all_files = os.listdir(cls_path)
    images = [f for f in all_files if f.lower().endswith(supported_formats)]

    if not images:
        print(f"Warning: No images found in {cls}")
        continue

    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    print(f"\nClass: {cls}")
    print(f"  Total: {n_total} | Train: {n_train} | Val: {n_val} | Test: {n_test}")

    for split_name, split_images in splits.items():
        split_cls_dir = os.path.join(output_dir, split_name, cls)
        os.makedirs(split_cls_dir, exist_ok=True)

        for img in split_images:
            src_path = os.path.join(cls_path, img)
            dst_path = os.path.join(split_cls_dir, img)

            if enable_cropping:
                success = crop_image(src_path, dst_path, crop_params)
                if success:
                    cropped_images += 1
                else:
                    failed_crops += 1
            else:
                shutil.copy2(src_path, dst_path)

            total_images += 1

# ----------------------------
# SUMMARY
# ----------------------------
print("\n" + "=" * 50)
print("DATASET SPLIT COMPLETED!")
print("=" * 50)
print(f"Total images processed: {total_images}")

if enable_cropping:
    print(f"Successfully cropped: {cropped_images}")
    print(f"Failed crops (used original): {failed_crops}")
    print(f"Crop success rate: {cropped_images / total_images * 100:.1f}%")

print(f"\nDataset structure created in: {output_dir}")
print("Directory structure:")
for split in ["train", "val", "test"]:
    split_path = os.path.join(output_dir, split)
    if os.path.exists(split_path):
        class_counts = []
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            if os.path.exists(cls_path):
                count = len([f for f in os.listdir(cls_path)
                             if f.lower().endswith(supported_formats)])
                class_counts.append(count)
        print(f"  {split}: {sum(class_counts)} images across {len([c for c in class_counts if c > 0])} classes")

