import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from datetime import datetime
import time
import json

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()


class BrickDataset(Dataset):
    """Dataset for pre-augmented brick recognition data"""

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.classes = [
            'white_1x3_good', 'white_2x2_good', 'white_2x4_good',
            'blue_2x2_good', 'blue_2x6_good', 'blue_1x6_good',
            'white_1x3_damaged', 'white_2x2_damaged', 'white_2x4_damaged',
            'blue_2x2_damaged', 'blue_2x6_damaged', 'blue_1x6_damaged',
            'no_brick'
        ]

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        self._load_dataset()

    def _load_dataset(self):
        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Dataset directory not found: {split_dir}")

        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

                for img_file in image_files:
                    self.images.append(os.path.join(class_dir, img_file))
                    self.labels.append(self.class_to_idx[class_name])

        print(f"{self.split}: {len(self.images)} images loaded")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


class MobileNetBrick(nn.Module):
    """MobileNet-V3-Small backbone with Pi-optimized head"""

    def __init__(self, num_classes=13, pretrained=True):
        super(MobileNetBrick, self).__init__()

        # Load MobileNet-V3-Small backbone
        if pretrained:
            from torchvision.models import MobileNet_V3_Small_Weights
            backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            backbone = models.mobilenet_v3_small(weights=None)

        # Extract only the feature extractor (remove classifier and avgpool)
        self.features = backbone.features  # Output: (batch, 576, 7, 7)

        # Add brick-specific adaptation layers (2 conv layers like Keras)
        self.brick_adapter = nn.Sequential(
            # First conv: moderate channel reduction
            nn.Conv2d(576, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Second conv: feature refinement
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Pi-optimized classifier with gradual reduction
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Extract backbone features
        x = self.features(x)  # (batch, 576, 7, 7)

        # Brick-specific adaptation
        x = self.brick_adapter(x)  # (batch, 32, 7, 7)

        # Global pooling
        x = self.global_pool(x)  # (batch, 32, 1, 1)

        # Classification
        x = self.classifier(x)  # (batch, num_classes)

        return x

    def freeze_backbone(self, freeze=True):
        """Freeze backbone feature layers for fine-tuning"""
        for param in self.features.parameters():
            param.requires_grad = not freeze


def load_config(config_path='train_config.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Remove description field if it exists (it's just for documentation)
        if 'description' in config:
            del config['description']

        # Handle null values for load_checkpoint
        if config.get('load_checkpoint') == 'null' or config.get('load_checkpoint') == '':
            config['load_checkpoint'] = None

        return config
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found. Using default configuration.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return None


def get_transforms(split='train'):
    if split == 'train':
        # Heavy augmentation for training only
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Clean transforms for val/test
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced datasets"""
    class_counts = {}
    for label in dataset.labels:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    total_samples = len(dataset.labels)
    num_classes = len(dataset.classes)

    class_weights = []
    for class_name in dataset.classes:
        count = class_counts.get(class_name, 1)
        weight = total_samples / (num_classes * count)
        class_weights.append(weight)

    return torch.FloatTensor(class_weights)


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001,
                patience=15, class_weights=None):
    """Training loop with early stopping"""

    # Setup training
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)

    # Training tracking
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        # Update scheduler
        scheduler.step(val_acc)

        # Record history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            flag = "NEW BEST"
        else:
            patience_counter += 1
            flag = f"({patience_counter}/{patience})"

        print(f'Epoch {epoch + 1:3d}: Train[{train_acc:5.1f}%, {train_loss:.3f}] '
              f'Val[{val_acc:5.1f}%, {val_loss:.3f}] {flag}')

        if patience_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    total_time = time.time() - start_time
    print(f'Training completed in {total_time / 60:.1f} minutes')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')

    history['best_val_acc'] = best_val_acc
    history['total_epochs'] = epoch + 1
    return model, history


def evaluate_model(model, test_loader, class_names):
    """Evaluate model on test set with detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Calculate precision, recall, F1-score for each class
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None,
                                                                     zero_division=0)

    # Check which classes were never predicted
    unique_predicted = set(all_preds)
    unique_actual = set(all_labels)
    never_predicted = unique_actual - unique_predicted

    if never_predicted:
        print(f"\nWARNING: Model never predicted these classes: {[class_names[i] for i in never_predicted]}")
        print("This suggests the model may be overly confident or these classes are very difficult to distinguish.")

    # Print detailed per-class metrics
    print("\nDetailed Per-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 80)

    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<8}")

    # Highlight damaged brick performance for quality control
    print("\nDAMAGED BRICK ANALYSIS (Quality Control Focus):")
    print("-" * 60)
    damaged_classes = [i for i, name in enumerate(class_names) if 'damaged' in name]

    if damaged_classes:
        print("Recall for damaged bricks (ability to catch damaged items):")
        for i in damaged_classes:
            print(f"  {class_names[i]:<25}: {recall[i]:.3f} ({recall[i] * 100:.1f}%)")

        avg_damaged_recall = np.mean([recall[i] for i in damaged_classes])
        print(f"\nAverage damaged brick recall: {avg_damaged_recall:.3f} ({avg_damaged_recall * 100:.1f}%)")

        if avg_damaged_recall < 0.95:
            print("WARNING: Damaged brick detection below 95%. Consider model tuning for quality control.")
        else:
            print("Damaged brick detection above 95% - good for quality control.")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, cm


def plot_results(history, cm, class_names):
    """Plot training curves and confusion matrix"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history['train_acc']) + 1)

    # Training curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation')
    ax2.axhline(y=history['best_val_acc'], color='g', linestyle='--',
                label=f'Best: {history["best_val_acc"]:.1f}%')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=class_names, yticklabels=class_names)
    ax3.set_title('Confusion Matrix')
    ax3.tick_params(axis='x', rotation=45)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    ax4.bar(range(len(class_names)), per_class_acc)
    ax4.set_title('Per-Class Accuracy')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_xticks(range(len(class_names)))
    ax4.set_xticklabels(class_names, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def load_model_checkpoint(checkpoint_path, device):
    """Load a saved model checkpoint for continued training"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Recreate model with same architecture
    model = MobileNetBrick(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Loaded model with {checkpoint['best_val_accuracy']:.2f}% validation accuracy")
    print(f"Training was completed on: {checkpoint['training_completed']}")

    return model, checkpoint


def save_torchscript_model(model, path='brick_classifier_torchscript.pt'):
    """Save model as TorchScript for easy deployment"""
    model.eval()  # Important: set to evaluation mode

    try:
        # Try scripting first (preferred method)
        print("Attempting TorchScript scripting...")
        scripted_model = torch.jit.script(model)
        scripted_model.save(path)
        print(f"TorchScript model saved to {path}")
        print(f"Load with: model = torch.jit.load('{path}')")
        return True
    except Exception as e:
        print(f"Scripting failed: {e}")
        print("Trying trace method...")

        try:
            # Fallback to tracing with dummy input tensor
            dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(path)
            print(f"TorchScript (traced) model saved to {path}")
            print(f"Load with: model = torch.jit.load('{path}')")
            return True
        except Exception as e2:
            print(f"Both scripting and tracing failed: {e2}")
            return False


def save_model(model, class_names, history, config, path='brick_classifier.pth', save_torchscript=True):
    """Save trained model with complete configuration for easy loading"""
    model_info = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names),
        'best_val_accuracy': history['best_val_acc'],
        'training_config': config,
        'model_architecture': {
            'class_name': 'MobileNetBrick',
            'num_classes': len(class_names),
            'pretrained': True,
            'backbone': 'mobilenet_v3_small',
            'input_size': (224, 224),
            'backbone_output': (576, 7, 7),
            'brick_adapter': {
                'conv1': {'in_channels': 576, 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
                'batch_norm1': True,
                'activation1': 'ReLU',
                'conv2': {'in_channels': 64, 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
                'batch_norm2': True,
                'activation2': 'ReLU'
            },
            'global_pool': {'type': 'AdaptiveAvgPool2d', 'output_size': (1, 1)},
            'classifier': {
                'layer_sizes': [32, 256, 128, 64, 32, len(class_names)],
                'dropout_rates': [0.5, 0.4, 0.3, 0.2],
                'activation': 'ReLU'
            },
            'total_flow': '(576,7,7) -> adapter -> (32,7,7) -> pool -> (32,) -> 256 -> 128 -> 64 -> 32 -> (13,)'
        },
        'transforms': {
            'train': {
                'RandomRotation': {'degrees': 15},
                'RandomAffine': {'degrees': 0, 'translate': (0.1, 0.1), 'scale': (0.9, 1.1), 'shear': 5},
                'ColorJitter': {'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.2},
                'RandomHorizontalFlip': {'p': 0.3},
                'Resize': {'size': (224, 224)},
                'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
            },
            'inference': {
                'Resize': {'size': (224, 224)},
                'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
            }
        },
        'training_history': history,
        'training_completed': datetime.now().isoformat(),
        'pytorch_version': torch.__version__
    }

    # Save standard PyTorch model
    torch.save(model_info, path)
    print(f"Model saved to {path} with complete architecture and configuration")

    # Save TorchScript version if requested
    if save_torchscript:
        torchscript_path = path.replace('.pth', '_torchscript.pt')
        success = save_torchscript_model(model, torchscript_path)
        if success:
            print("TorchScript version can be loaded without the MobileNetBrick class definition!")
        else:
            print("TorchScript saving failed - only standard .pth model available")

    print("Your friend can load this model without needing to know any internal parameters!")


def main():
    """Main training pipeline with MobileNetBrick and TorchScript support"""
    print("=" * 60)
    print("BRICK RECOGNITION TRAINING - MOBILENET WITH TORCHSCRIPT")
    print("=" * 60)

    # Load configuration from JSON file
    config = load_config('train_config.json')

    # Fallback to default configuration if loading fails
    if config is None:
        config = {
            'dataset_path': 'dataset_split',
            'batch_size': 32,
            'num_epochs': 8,
            'learning_rate': 0.0005,
            'patience': 15,
            'use_class_weights': True,
            'freeze_backbone': False,
            'load_checkpoint': 'brick_classifier_simple.pth',
            'save_torchscript': True,
        }
        print("Using default configuration")

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")


    x = 1
    if x == 1:
        return

    # Load datasets
    try:
        train_dataset = BrickDataset(config['dataset_path'], 'train', get_transforms('train'))
        val_dataset = BrickDataset(config['dataset_path'], 'val', get_transforms('val'))
        test_dataset = BrickDataset(config['dataset_path'], 'test', get_transforms('test'))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure your dataset directory exists!")
        return

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Model setup - with checkpoint loading capability
    if config['load_checkpoint']:
        # ================= CONTINUED TRAINING MODE =================
        model, checkpoint = load_model_checkpoint(config['load_checkpoint'], device)
        model.freeze_backbone(freeze=config['freeze_backbone'])
        print(f"Resumed training - Backbone frozen: {config['freeze_backbone']}")
    else:
        # ================= FRESH TRAINING MODE =================
        model = MobileNetBrick(num_classes=len(train_dataset.classes))
        model = model.to(device)
        model.freeze_backbone(freeze=config['freeze_backbone'])
        print(f"Fresh training - Using: MobileNet-V3-Small with ImageNet pretraining")
        print(f"Backbone frozen: {config['freeze_backbone']}")

    # Adjust learning rate for unfrozen fine-tuning
    if not config['freeze_backbone']:
        config['learning_rate'] = config['learning_rate'] * 0.1  # Lower LR for fine-tuning
        print(f"Fine-tuning mode: Reduced learning rate to {config['learning_rate']}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Class weights
    class_weights = None
    if config['use_class_weights']:
        class_weights = calculate_class_weights(train_dataset)
        print("Using class weights for balanced training")

    # Training
    model, history = train_model(
        model, train_loader, val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        class_weights=class_weights
    )

    # Evaluation
    test_accuracy, cm = evaluate_model(model, test_loader, train_dataset.classes)

    # Results and saving
    plot_results(history, cm, train_dataset.classes)
    save_model(model, train_dataset.classes, history, config,
               'brick_classifier_simple.pth',
               save_torchscript=config['save_torchscript'])

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Best Val Accuracy: {history['best_val_acc']:.2f}%")
    print(f"Total Epochs: {history['total_epochs']}")

    files_saved = ["brick_classifier_simple.pth", "training_results.png"]
    if config['save_torchscript']:
        files_saved.append("brick_classifier_simple_torchscript.pt")

    print(f"Files saved: {', '.join(files_saved)}")
    print("=" * 60)


if __name__ == "__main__":
    main()