import os
import random
import shutil
from datetime import datetime

# Brick types for condition detection (same as Raspberry Pi script)
brick_types = [
    # Good condition bricks
    "white_1x3_good", "white_2x2_good", "white_2x4_good",
    "blue_2x2_good", "blue_2x6_good", "blue_1x6_good",
    # Damaged condition bricks
    "white_1x3_damaged", "white_2x2_damaged", "white_2x4_damaged",
    "blue_2x2_damaged", "blue_2x6_damaged", "blue_1x6_damaged",
    # Background
    "no_brick"
]

# Create train/val/test folders if they don't exist
train_val_test_folders = []
for brick_type in brick_types:
    train_val_test_folders.extend([
        f"brick_dataset/train/{brick_type}",
        f"brick_dataset/val/{brick_type}",
        f"brick_dataset/test/{brick_type}"
    ])

for folder in train_val_test_folders:
    os.makedirs(folder, exist_ok=True)


def clear_train_val_test_folders():
    """Clear existing train/val/test folders before splitting"""
    print("Clearing existing train/val/test folders...")
    for brick_type in brick_types:
        for split in ["train", "val", "test"]:
            folder = f"brick_dataset/{split}/{brick_type}"
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
    print("✓ Cleared existing files")


def split_dataset():
    """Split raw + augmented images into train/val/test folders"""

    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    print(f"\nDataset Split Configuration:")
    print(f"  Train: {train_ratio * 100:.0f}%")
    print(f"  Val:   {val_ratio * 100:.0f}%")
    print(f"  Test:  {test_ratio * 100:.0f}%")
    print("-" * 50)

    total_images_processed = 0

    for brick_type in brick_types:
        print(f"Processing {brick_type}...")

        # Collect all images for this brick type
        raw_folder = f"brick_dataset/raw/{brick_type}"
        augmented_folder = f"brick_dataset/augmented/{brick_type}"

        all_images = []
        raw_count = 0
        augmented_count = 0

        # Add raw images
        if os.path.exists(raw_folder):
            for img_file in os.listdir(raw_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(raw_folder, img_file))
                    raw_count += 1

        # Add augmented images
        if os.path.exists(augmented_folder):
            for img_file in os.listdir(augmented_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(augmented_folder, img_file))
                    augmented_count += 1

        if not all_images:
            print(f"  ⚠️  No images found for {brick_type}")
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
        def copy_images(image_list, dest_folder, split_name):
            copied = 0
            for img_path in image_list:
                try:
                    filename = os.path.basename(img_path)
                    dest_path = os.path.join(dest_folder, filename)
                    shutil.copy2(img_path, dest_path)
                    copied += 1
                except Exception as e:
                    print(f"    ✗ Failed to copy {img_path}: {e}")
            return copied

        # Copy to train/val/test folders
        train_copied = copy_images(train_images, f"brick_dataset/train/{brick_type}", "train")
        val_copied = copy_images(val_images, f"brick_dataset/val/{brick_type}", "val")
        test_copied = copy_images(test_images, f"brick_dataset/test/{brick_type}", "test")

        print(f"  📊 Raw: {raw_count}, Augmented: {augmented_count}, Total: {total_images}")
        print(f"  📁 Train: {train_copied}, Val: {val_copied}, Test: {test_copied}")

        total_images_processed += total_images

    print("-" * 50)
    print(f"✅ Dataset splitting completed!")
    print(f"📈 Total images processed: {total_images_processed}")

    return total_images_processed


def show_dataset_summary():
    """Show summary of the split dataset"""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    total_train = total_val = total_test = 0

    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()} SET:")
        split_total = 0

        for brick_type in brick_types:
            folder = f"brick_dataset/{split}/{brick_type}"
            if os.path.exists(folder):
                count = len([f for f in os.listdir(folder)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {brick_type:18}: {count:4d} images")
                split_total += count
            else:
                print(f"  {brick_type:18}: {0:4d} images")

        print(f"  {'TOTAL':18}: {split_total:4d} images")

        if split == "train":
            total_train = split_total
        elif split == "val":
            total_val = split_total
        elif split == "test":
            total_test = split_total

    print("-" * 60)
    print(f"GRAND TOTAL: {total_train + total_val + total_test} images")
    print(f"Split: {total_train} train / {total_val} val / {total_test} test")
    print("=" * 60)


def main():
    print("🧱 LEGO Brick Dataset Splitter for Windows")
    print("=" * 60)
    print("This script splits your Raspberry Pi collected data into train/val/test")
    print("Make sure you have:")
    print("  - brick_dataset/raw/ folder (from Raspberry Pi)")
    print("  - brick_dataset/augmented/ folder (from Raspberry Pi)")
    print("=" * 60)

    # Check if source folders exist
    if not os.path.exists("brick_dataset"):
        print("❌ ERROR: brick_dataset folder not found!")
        print("   Make sure you copied the dataset from your Raspberry Pi")
        return

    if not os.path.exists("brick_dataset/raw"):
        print("❌ ERROR: brick_dataset/raw folder not found!")
        return

    # Show current status
    print("\nCurrent dataset status:")
    raw_count = 0
    augmented_count = 0

    for brick_type in brick_types:
        raw_folder = f"brick_dataset/raw/{brick_type}"
        augmented_folder = f"brick_dataset/augmented/{brick_type}"

        if os.path.exists(raw_folder):
            raw_count += len([f for f in os.listdir(raw_folder)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if os.path.exists(augmented_folder):
            augmented_count += len([f for f in os.listdir(augmented_folder)
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"  Raw images: {raw_count}")
    print(f"  Augmented images: {augmented_count}")
    print(f"  Total images to split: {raw_count + augmented_count}")

    if raw_count + augmented_count == 0:
        print("❌ No images found to split!")
        return

    # Confirm before proceeding
    print("\n⚠️  This will overwrite existing train/val/test folders")
    confirm = input("Continue? (y/N): ").lower().strip()

    if confirm != 'y':
        print("Cancelled.")
        return

    # Set random seed for reproducible splits
    random.seed(42)
    print(f"🎲 Using random seed: 42 (for reproducible splits)")

    # Clear existing splits
    clear_train_val_test_folders()

    # Split the dataset
    total_processed = split_dataset()

    # Show summary
    show_dataset_summary()

    # Save split info
    split_info = {
        "timestamp": datetime.now().isoformat(),
        "total_images": total_processed,
        "split_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
        "random_seed": 42,
        "brick_types": brick_types
    }

    import json
    with open("brick_dataset/split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n✅ Dataset ready for training!")
    print(f"📄 Split info saved to: brick_dataset/split_info.json")
    print("\nNext steps:")
    print("  1. Use brick_dataset/train/ for training")
    print("  2. Use brick_dataset/val/ for validation")
    print("  3. Use brick_dataset/test/ for final testing")


if __name__ == "__main__":
    main()