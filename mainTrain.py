import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import cv2
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check for CUDA availability
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
else:
    print("CUDA not available, using CPU")


class BrickDataset(Dataset):
    """Custom dataset for LEGO brick recognition and condition detection"""

    def __init__(self, root_dir, split='train', transform=None, use_augmented=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.use_augmented = use_augmented

        # Define all classes based on your collection script
        self.classes = [
            'white_1x3_good', 'white_2x2_good', 'white_2x4_good',
            'blue_2x2_good', 'blue_2x6_good', 'blue_1x6_good',
            'white_1x3_damaged', 'white_2x2_damaged', 'white_2x4_damaged',
            'blue_2x2_damaged', 'blue_2x6_damaged', 'blue_1x6_damaged',
            'no_brick'
        ]

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Load image paths and labels
        self.images = []
        self.labels = []

        # Your collection script already populates train/val/test with both raw and augmented images
        # so we just need to load from the split directories
        split_dir = os.path.join(root_dir, split)

        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])

        print(f"{split.upper()} dataset: {len(self.images)} images across {len(self.classes)} classes")
        print(f"  (Includes pre-augmented data from your Albumentations pipeline)")

        # Print class distribution
        class_counts = {}
        raw_counts = {}
        aug_counts = {}

        for i, label in enumerate(self.labels):
            class_name = self.idx_to_class[label]
            img_path = self.images[i]

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Check if it's an augmented image (based on filename from your script)
            if 'aug_' in os.path.basename(img_path):
                aug_counts[class_name] = aug_counts.get(class_name, 0) + 1
            else:
                raw_counts[class_name] = raw_counts.get(class_name, 0) + 1

        for class_name in self.classes:
            total = class_counts.get(class_name, 0)
            raw = raw_counts.get(class_name, 0)
            aug = aug_counts.get(class_name, 0)
            print(f"  {class_name}: {total} total ({raw} raw + {aug} augmented)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


class BrickNet(nn.Module):
    """CNN optimized for LEGO brick recognition"""

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


def get_transforms(image_size=224):
    """Get minimal transforms - only normalization since augmentation is already done"""

    # Only basic preprocessing since your Albumentations pipeline already handles augmentation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Same transform for train/val/test since augmentation is pre-applied
    return transform, transform


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Training loop with validation"""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_state = None

    print("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Print progress for long training on Pi
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
              f'Time: {epoch_time:.1f}s')

    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time:.1f}s')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')

    # Load best model
    model.load_state_dict(best_model_state)

    return model, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }


def evaluate_model(model, test_loader, class_names):
    """Evaluate model and generate detailed metrics"""

    model.eval()
    all_preds = []
    all_labels = []

    print("Evaluating model...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, cm


def plot_training_history(history):
    """Plot training curves"""

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracies'], label='Train Acc')
    plt.plot(history['val_accuracies'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_model_for_inference(model, class_names, model_path='brick_classifier.pth'):
    """Save model with metadata for inference"""

    model_info = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names),
        'timestamp': datetime.now().isoformat(),
        'input_size': (224, 224),
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }

    torch.save(model_info, model_path)
    print(f"Model saved to {model_path}")


def main():
    """Main training pipeline"""

    print("=" * 70)
    print("LEGO BRICK RECOGNITION & CONDITION DETECTION TRAINING")
    print("=" * 70)

    # Configuration for Windows training
    dataset_path = 'brick_dataset'
    batch_size = 32  # Can use larger batch size on Windows
    num_epochs = 50  # Full training epochs
    learning_rate = 0.001
    image_size = 224

    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Image size: {image_size}x{image_size}")

    # Get transforms (minimal since augmentation already done by your collection script)
    train_transform, val_transform = get_transforms(image_size)

    # Create datasets
    train_dataset = BrickDataset(dataset_path, 'train', train_transform)
    val_dataset = BrickDataset(dataset_path, 'val', val_transform)
    test_dataset = BrickDataset(dataset_path, 'test', val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Create model
    model = BrickNet(num_classes=len(train_dataset.classes), use_pretrained=True)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    model, history = train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    # Evaluate on test set
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)

    test_accuracy, confusion_mat = evaluate_model(model, test_loader, train_dataset.classes)

    # Plot results
    print("\nGenerating plots...")
    try:
        plot_training_history(history)
        plot_confusion_matrix(confusion_mat, train_dataset.classes)
    except Exception as e:
        print(f"Error generating plots: {e}")

    # Save model
    save_model_for_inference(model, train_dataset.classes)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Best Validation Accuracy: {history['best_val_acc']:.2f}%")
    print("Model saved as 'brick_classifier.pth'")
    print("=" * 70)

    # Save training summary
    summary = {
        'test_accuracy': test_accuracy,
        'best_val_accuracy': history['best_val_acc'],
        'total_epochs': num_epochs,
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        },
        'classes': train_dataset.classes,
        'training_time': datetime.now().isoformat()
    }

    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("Training summary saved to 'training_summary.json'")


if __name__ == "__main__":
    main()