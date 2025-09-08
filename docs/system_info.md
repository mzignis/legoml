# Project Documentation - LEGO Brick Detection System using AI

## 1. Introduction

### Project Description
The "LegoML" project is a LEGO brick detection and classification system using machine learning techniques and neural networks. The system is capable of automatically recognizing different types of LEGO bricks based on image analysis.

**Main Objective:**
- Automatic recognition and classification of different types of LEGO bricks
- Implementation of an efficient computer vision system for small objects
- Demonstration of modern deep learning techniques in practical applications

**Brief System Description:**
The system uses a camera to capture images of LEGO bricks on a conveyor belt. Images are then processed by a pre-trained neural network that classifies the brick type. The system can operate in real-time mode or process images in batches.

### Project Scope
The system is capable of detecting the following objects/bricks:
- LEGO Bricks 1x6 Blue (Good/Damaged)
- LEGO Bricks 2x2 Blue (Good/Damaged)
- LEGO Bricks 2x6 Blue (Good/Damaged)
- LEGO Bricks 1x3 White (Good/Damaged)
- LEGO Bricks 2x2 White (Good/Damaged)
- LEGO Bricks 2x4 White (Good/Damaged)
- No Brick

## 2. Key System Components

### 2.1 Hardware

#### LEGO Elements
- **Motors:**
  - LEGO SPIKE Prime Hub
  - LEGO Large Motor
  - LEGO Medium Motor x2

- **Construction Elements:**
  - LEGO Technic Beams
  - Connectors and Axles
  - Mounting Elements

```
Raspberry Pi 4B
    └── Camera
LEGO Hub Controller
    ├── LEGO Large Motor
    └── LEGO Medium Motor x2
```

#### Raspberry Pi
- **Model:** Raspberry Pi 4B
- **Operating System Version:** Raspberry Pi OS Bookworm (12)
- **RAM:** 8GB
- **Other Important Information:**
  - Camera Module 2, Samsung EVO Plus 64GB microSD

### 2.2 Communication Between Components

[Communication details to be added]

## 3. Software

### 3.1 Environment and Installation

#### Python / MicroPython
- **Python Version:** Python 3
- **Installation Method:**
```bash
# Specific installation commands
sudo apt update
sudo apt install python3 python3-pip
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mzignis/legoml.git
   cd legoml
   ```

2. **Install dependencies**

   ```bash
   #rPi
   pip install -r requirementsRpi.txt
   ```
   ```bash
   # Install rPi system dependencies
   sudo apt update
   sudo apt install python3-opencv libatlas-base-dev python3-picamera2
   ```
   ```bash
   #Windows
   pip install -r requirementsWin.txt
   ```

#### Communication Software
**Main Libraries:**
- **OpenCV** - Image processing
- **PyTorch** - Neural network
- **NumPy** - Matrix and array operations
- **Matplotlib** - Data visualization

## 4. Neural Network / AI

### 4.1 Training Description

## 🔧 Configuration

Edit `train_config.json` `split_config.json`

#### Data Collection
- **Method:** Photos on conveyor belt
- **Number of Images:** 40 per class
- **Categories:** 13

#### Data Preprocessing
```python
# Example preprocessing

# Before training script dataSplit.py
"cropping_config": {
    "enable_cropping": true,
    "crop_params": {
      "left": 164,
      "top": 164,
      "right": 164,
      "bottom": 164
    }
}

# Dynamically during training
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
```

#### Training Details
- **Network Architecture:** PyTorch NN
```python
# backbone - mobilenet_v3_small
self.features = backbone.features  # Output: (batch, 576, 7, 7)

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

self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

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
```

- **Parameters (train_config.json):**
  - Learning rate: 0.005
  - Batch size: 32
  - Epochs: 75
  - Patience: 15
  - Optimizer: Adam

#### Running Training on New Data
```bash
python dataSplit.py
python train_model.py
```

### 4.2 Data Description

#### Data Types
- **Image Format:** JPG
- **Resolution:** 1024x768
- **Folder Structure:**
```
data/
├── train/
│   ├── blue_1x6_damaged/
│   ├── blue_1x6_good/
│   ...
└── test/
    ...
```

## 5. References and Sources

### Libraries and Tools
- **[OpenCV](https://opencv.org/)** - Image processing and camera handling
- **[PyTorch](https://pytorch.org/)** - Machine learning framework
- **[NumPy](https://numpy.org/)** - Matrix and array operations
- **[Matplotlib](https://matplotlib.org/)** - Data and results visualization

### Technical Documentation
- **[Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)** - Official venv documentation
- **[OpenCV Python Tutorials](https://opencv-python-tutroals.readthedocs.io/)** - OpenCV tutorials

---

## License
Project available under the license specified in the GitHub repository.

---