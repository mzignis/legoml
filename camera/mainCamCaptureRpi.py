from picamera2 import Picamera2
import cv2
import os
import json
from datetime import datetime
import albumentations as A

# Create dataset structure for your specific bricks
dataset_folders = [
    # Raw images (backup)
    "brick_dataset/raw/white_1x3",
    "brick_dataset/raw/white_2x2",
    "brick_dataset/raw/white_2x4",
    "brick_dataset/raw/blue_2x2",
    "brick_dataset/raw/blue_2x6",
    "brick_dataset/raw/blue_1x6",
    "brick_dataset/raw/no_brick",
    # Training structure
    "brick_dataset/train/white_1x3",
    "brick_dataset/train/white_2x2",
    "brick_dataset/train/white_2x4",
    "brick_dataset/train/blue_2x2",
    "brick_dataset/train/blue_2x6",
    "brick_dataset/train/blue_1x6",
    "brick_dataset/train/no_brick",
    # Validation structure
    "brick_dataset/val/white_1x3",
    "brick_dataset/val/white_2x2",
    "brick_dataset/val/white_2x4",
    "brick_dataset/val/blue_2x2",
    "brick_dataset/val/blue_2x6",
    "brick_dataset/val/blue_1x6",
    "brick_dataset/val/no_brick",
    # Test structure
    "brick_dataset/test/white_1x3",
    "brick_dataset/test/white_2x2",
    "brick_dataset/test/white_2x4",
    "brick_dataset/test/blue_2x2",
    "brick_dataset/test/blue_2x6",
    "brick_dataset/test/blue_1x6",
    "brick_dataset/test/no_brick",
    # Augmented folders
    "brick_dataset/augmented/white_1x3",
    "brick_dataset/augmented/white_2x2",
    "brick_dataset/augmented/white_2x4",
    "brick_dataset/augmented/blue_2x2",
    "brick_dataset/augmented/blue_2x6",
    "brick_dataset/augmented/blue_1x6",
    "brick_dataset/augmented/no_brick"
]

for folder in dataset_folders:
    os.makedirs(folder, exist_ok=True)

# Albumentations pipeline optimized for conveyor belt brick recognition
augmentation_pipeline = A.Compose([
    A.RandomRotate90(p=0.8),  # Very common - bricks tumble and land in any 90° orientation
    A.Rotate(limit=5, p=0.6),  # Small angle variations as bricks shift on belt
    A.HorizontalFlip(p=0.5),  # Bricks can be upside down
    A.VerticalFlip(p=0.5),  # Bricks can be upside down
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),  # Lighting variations along belt
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5),  # Color variations
    A.RandomShadow(p=0.4),  # Belt lighting creates shadows
    A.GaussNoise(noise_scale_factor=0.9, per_channel=True, p=0.3),  # Camera noise
])

# Camera setup
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

# Current brick type being collected - Updated for your specific bricks
brick_types = [
    "white_1x3", "white_2x2", "white_2x4",  # White bricks
    "blue_2x2", "blue_2x6", "blue_1x6",  # Blue bricks
    "no_brick"  # Background/empty
]
current_type_index = 0
current_type = brick_types[current_type_index]

# Counters
image_counters = {brick_type: 0 for brick_type in brick_types}
total_images = 0

# Metadata storage
metadata = {
    "dataset_info": {
        "created": datetime.now().isoformat(),
        "camera": "Raspberry Pi Camera Module 2",
        "resolution": "640x480",
        "purpose": "LEGO Brick Recognition Training Data",
        "platform": "Raspberry Pi"
    },
    "classes": {},
    "augmentation_config": str(augmentation_pipeline)
}

print("LEGO Brick Recognition Data Collection System")
print("=" * 60)
print("Brick Types to Collect:")
print("  White: 1x3, 2x2, 2x4")
print("  Blue:  2x2, 2x6, 1x6")
print("  Plus:  no_brick (empty/background)")
print("=" * 60)
print("Controls:")
print("'q' - Quit and save metadata")
print("'n' - Switch to next brick type")
print("'p' - Switch to previous brick type")
print("'s' - Show current statistics")
print("Space - Capture image")
print("'a' - Auto-augment last captured image (5x)")
print("'r' - Reset current type counter")
print("'d' - Auto-populate train/val/test dataset")
print("=" * 60)


def save_metadata():
    """Save collection metadata"""
    for brick_type in brick_types:
        metadata["classes"][brick_type] = {
            "count": image_counters[brick_type],
            "folder": f"brick_dataset/raw/{brick_type}"
        }

    with open("brick_dataset/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to brick_dataset/metadata.json")


def augment_image(image_path, brick_type, base_name, count=5):
    """Apply augmentations to an image and save in organized folders"""
    try:
        print(f"Starting augmentation of {image_path}")

        # Check if source image exists
        if not os.path.exists(image_path):
            print(f"ERROR: Source image {image_path} does not exist!")
            return

        # Read and convert image
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: Could not read image {image_path}")
            return

        print(f"Image loaded successfully, shape: {image.shape}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # HERE - "augmented" prefix is added to the path
        output_folder = f"brick_dataset/augmented/{brick_type}"
        print(f"Saving augmented images to: {output_folder}")

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        for i in range(count):
            print(f"Generating augmentation {i + 1}/{count}")

            # Apply augmentation
            augmented = augmentation_pipeline(image=image_rgb)
            augmented_image = augmented['image']

            # Convert back to BGR for saving
            augmented_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            # Save augmented image
            aug_filename = f"{output_folder}/aug_{base_name}_{i + 1:02d}.jpg"
            print(f"Saving: {aug_filename}")

            success = cv2.imwrite(aug_filename, augmented_bgr)
            if success:
                print(f"  ✓ Saved successfully")
            else:
                print(f"  ✗ Failed to save {aug_filename}")

        print(f"Augmentation completed for {brick_type}")

    except Exception as e:
        print(f"ERROR in augment_image: {e}")
        import traceback
        traceback.print_exc()


def auto_populate_dataset():
    """Automatically populate train/val/test folders from raw + augmented images"""
    import random
    import shutil

    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    for brick_type in brick_types:
        print(f"Processing {brick_type}...")

        # Collect all images for this brick type
        raw_folder = f"brick_dataset/raw/{brick_type}"
        augmented_folder = f"brick_dataset/augmented/{brick_type}"

        all_images = []

        # Add raw images
        if os.path.exists(raw_folder):
            for img_file in os.listdir(raw_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(raw_folder, img_file))

        # Add augmented images for this brick type
        if os.path.exists(augmented_folder):
            for img_file in os.listdir(augmented_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(augmented_folder, img_file))

        if not all_images:
            print(f"  No images found for {brick_type}")
            continue

        # Shuffle images for random split
        random.shuffle(all_images)

        # Calculate split indices
        total_images = len(all_images)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)

        # Split images
        train_images = all_images[:train_count]
        val_images = all_images[train_count:train_count + val_count]
        test_images = all_images[train_count + val_count:]

        # Copy images to respective folders
        def copy_images(image_list, dest_folder):
            for img_path in image_list:
                filename = os.path.basename(img_path)
                dest_path = os.path.join(dest_folder, filename)
                shutil.copy2(img_path, dest_path)

        copy_images(train_images, f"brick_dataset/train/{brick_type}")
        copy_images(val_images, f"brick_dataset/val/{brick_type}")
        copy_images(test_images, f"brick_dataset/test/{brick_type}")

        print(f"  {brick_type}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    print("Dataset auto-population completed!")


def show_stats():
    """Display collection statistics"""
    print("\n" + "=" * 50)
    print("LEGO BRICK COLLECTION STATISTICS")
    print("=" * 50)

    # Group by color for better organization
    print("WHITE BRICKS:")
    for brick_type in ["white_1x3", "white_2x2", "white_2x4"]:
        print(f"  {brick_type:12}: {image_counters[brick_type]:4d} images")

    print("\nBLUE BRICKS:")
    for brick_type in ["blue_2x2", "blue_2x6", "blue_1x6"]:
        print(f"  {brick_type:12}: {image_counters[brick_type]:4d} images")

    print(f"\nBACKGROUND:")
    print(f"  {'no_brick':12}: {image_counters['no_brick']:4d} images")

    print("-" * 50)
    print(f"TOTAL RAW IMAGES: {sum(image_counters.values()):4d}")

    # Count augmented images by type
    augmented_counts = {}
    for brick_type in brick_types:
        augmented_folder = f"brick_dataset/augmented/{brick_type}"
        if os.path.exists(augmented_folder):
            augmented_counts[brick_type] = len([f for f in os.listdir(augmented_folder)
                                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            augmented_counts[brick_type] = 0

    total_augmented = sum(augmented_counts.values())
    print(f"AUGMENTED IMAGES: {total_augmented:4d}")
    print(f"TOTAL DATASET:    {sum(image_counters.values()) + total_augmented:4d}")

    # Show completion percentage (assuming target of ~50 images per type)
    target_per_type = 50
    completion = (sum(image_counters.values()) / (len(brick_types) * target_per_type)) * 100
    print(f"COMPLETION:       {completion:.1f}% (target: {target_per_type} per type)")
    print("=" * 50 + "\n")


def test_folder_access():
    """Test if all folders are accessible"""
    print("Testing folder access...")
    for brick_type in brick_types:
        folders_to_check = [
            f"brick_dataset/raw/{brick_type}",
            f"brick_dataset/augmented/{brick_type}",
            f"brick_dataset/train/{brick_type}",
            f"brick_dataset/val/{brick_type}",
            f"brick_dataset/test/{brick_type}"
        ]

        for folder in folders_to_check:
            exists = os.path.exists(folder)
            writable = os.access(folder, os.W_OK) if exists else "N/A"
            print(f"{folder}: exists={exists}, writable={writable}")


# Test folder access once at startup
test_folder_access()

while True:
    frame = picam2.capture_array()

    # Keep original frame clean for saving
    clean_frame = frame.copy()

    # Add better UI overlay with color coding (only for display)
    color = (0, 255, 0)  # Green for white bricks
    if "blue" in current_type:
        color = (255, 0, 0)  # Blue for blue bricks
    elif current_type == "no_brick":
        color = (128, 128, 128)  # Gray for no brick

    cv2.putText(frame, f"Collecting: {current_type.upper().replace('_', ' ')}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Count: {image_counters[current_type]}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Total: {sum(image_counters.values())}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Type {current_type_index + 1}/{len(brick_types)}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Instructions
    cv2.putText(frame, "SPACE=Capture, N=Next, P=Prev", (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, "A=Augment, S=Stats, R=Reset, D=Dataset, Q=Quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("LEGO Brick Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        save_metadata()
        break
    elif key == ord('n'):
        # Switch to next brick type
        current_type_index = (current_type_index + 1) % len(brick_types)
        current_type = brick_types[current_type_index]
        print(f"Switched to collecting: {current_type}")
    elif key == ord('p'):
        # Switch to previous brick type
        current_type_index = (current_type_index - 1) % len(brick_types)
        current_type = brick_types[current_type_index]
        print(f"Switched to collecting: {current_type}")
    elif key == ord('r'):
        # Reset current type counter
        old_count = image_counters[current_type]
        image_counters[current_type] = 0
        print(f"Reset {current_type} counter from {old_count} to 0")
    elif key == ord('d'):
        # Auto-populate train/val/test dataset
        print("Auto-populating train/val/test dataset...")
        auto_populate_dataset()
    elif key == ord('s'):
        show_stats()
    elif key == ord(' '):
        # Capture image (clean frame without UI text)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"brick_dataset/raw/{current_type}/{current_type}_{image_counters[current_type]:04d}_{timestamp}.jpg"
        cv2.imwrite(filename, clean_frame)  # Save clean frame without UI
        image_counters[current_type] += 1
        print(f"Captured: {filename}")

        # Store last captured image path for potential augmentation
        last_captured = filename

    elif key == ord('a'):
        # Augment last captured image with detailed debugging
        try:
            if 'last_captured' in locals():
                print(f"DEBUG: last_captured = {last_captured}")
                print(f"DEBUG: current_type = {current_type}")
                print(f"DEBUG: File exists = {os.path.exists(last_captured)}")

                base_name = os.path.splitext(os.path.basename(last_captured))[0]
                print(f"DEBUG: base_name = {base_name}")
                print(f"DEBUG: Expected output folder = brick_dataset/augmented/{current_type}")

                # Call augmentation function - "augmented" prefix added inside function
                augment_image(last_captured, current_type, base_name, 5)

                # Check if files were created
                output_folder = f"brick_dataset/augmented/{current_type}"
                if os.path.exists(output_folder):
                    files = os.listdir(output_folder)
                    print(f"DEBUG: Files in {output_folder}: {files}")
                else:
                    print(f"DEBUG: Output folder {output_folder} doesn't exist!")
            else:
                print("No image to augment. Capture an image first.")
        except Exception as e:
            print(f"Augmentation failed: {e}")
            import traceback

            traceback.print_exc()

cv2.destroyAllWindows()
picam2.close()

print("\nData collection completed!")
show_stats()
print("Next steps:")
print("1. Review captured images")
print("2. Run batch augmentation if needed")
print("3. Train your model with the dataset")
print("4. Deploy for real-time brick recognition!")