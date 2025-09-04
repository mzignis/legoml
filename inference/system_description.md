# LEGO Brick Sorting Inference System Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Component Descriptions](#component-descriptions)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

## System Overview

The LEGO Brick Sorting Inference System is an automated solution for classifying and sorting LEGO bricks using computer vision and machine learning. The system captures real-time images of bricks, classifies them using a trained neural network model, and controls physical hardware to sort them based on their condition and properties.

### Key Features

- **Real-time Classification**: Uses PyTorch-based neural network for brick classification
- **Multi-class Detection**: Supports classification of various brick types, colors, and conditions
- **Hardware Integration**: Controls conveyor belts and pusher mechanisms via Bluetooth
- **Web Dashboard**: Real-time monitoring and visualization through Streamlit interface
- **Automated Workflow**: Fully automated sorting process with minimal human intervention
- **Snapshot Management**: Automatic capture and storage of classified images

### Supported Classifications

The system classifies LEGO bricks into the following categories:
- **Colors**: White, Blue
- **Sizes**: 1x3, 2x2, 2x4, 1x6, 2x6
- **Conditions**: Good (undamaged), Damaged
- **Special**: No brick detected

## Architecture

### System Components

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Main Process      │    │  Dashboard Process  │    │  Monitor Process    │
│                     │    │                     │    │                     │
│ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │
││SortingOrchestrator││    │ │ Streamlit App   │ │    │ │ UpdateDetector  │ │
│ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │
│ ┌─────────────────┐ │    └─────────────────────┘    └─────────────────────┘
│ │ClassifierManager│ │              │                           │
│ └─────────────────┘ │              │                           │
│ ┌─────────────────┐ │              │                           │
││HardwareController ││              │                           │
│ └─────────────────┘ │              │                           │
│ ┌─────────────────┐ │              │                           │
│ │SortingStrategy  │ │              │                           │
│ └─────────────────┘ │              │                           │
└─────────────────────┘              │                           │
            │                        │                           │
            │                        │                           │
            ▼                        ▼                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Shared Resources                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │ dashboard_data  │  │   snapshots     │  │  .dashboard_update_     │ │
│  │     .json       │  │   directory     │  │      signal             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```
**Processes:**
- main process taking care of classificatio and motor control
- second process to make dashboard independant
- background process used to monitor for updates and signal when detecting them

### Communication Pipeline

```
┌─────────────┐                                    ┌─────────────┐
│             │    After processing raspberry pi   │             │
│ Raspberry   │--- sends appropriate signals to --→│ PyBricks    │
│     Pi      │    the hub                         │    Hub      │
│             │                                    │             │
│             │←--- Hub sends confirmation of ---- │             │
│             │     which motors it activated      │             │
└─────────────┘     (mostly for debugging)         └─────────────┘
       ▲                                                  │
       │                                                  │
       │ camera sends                                     │ PyBricks Hub
       │ images to                                        │ activates motors
       │ raspberry pi                                     │ corresponding to
       │                                                  │ received signals
       │                                                  ▼
┌─────────────┐                                    ┌─────────────┐
│             │                                    │             │
│   Camera    │                                    │   Motors    │
│             │                                    │             │
└─────────────┘                                    └─────────────┘
```
**Connection Types:**
- Comunication between the Raspberry Pi and PyBricks Hub occurs entirely over bluetooth
- The onnections between the camera and Raspberry Pi as well as the Hub and the motors are over cable

### Control Flow

```
main.py (Entry Point)
│
└── sorting_orchestrator.py (Central Coordinator)
    ├── hardware_controller.py (Bluetooth Communication)
    │   └── [Communicates with] hub_controller.py (LEGO Hub - MicroPython)
    │
    ├── classifier_manager.py (ML Management)
    │   └── brick_classifier.py (Image Classification Core)
    │
    ├── sorting_strategy.py (Timing Logic)
    │   └── [Uses] hardware_controller.py (Passed as dependency)
    │
    └── dashboard_connector.py (UI Management)
        ├── [Spawns Process] dashboard_v1.py (Streamlit UI)
        └── [Spawns Process] update_detector.py::monitor_process (File Monitoring)
```

### Directory Structure

```
project_root/
├── inference/
│       └── inference_system/
│           ├── brick_classifier.py
│           ├── sorting_orchestrator.py
│           ├── hardware_controller.py
│           ├── classifier_manager.py
│           ├── sorting_strategy.py
│           ├── dashboard_connector.py
│           ├── update_detector.py
│           ├── dashboard_v1.py
│           └── main.py
|
├── snapshots/  # Created automatically
├── dashboard_data.json  # Created automatically
└── brick_classifier_simple_torchscript.pt  # Your model file 
```
**Side note:**
'snapshots', 'dashboard_data.json' and 'brick_classifier_simple_torchscript.pt' all have
configurable filepaths in 'main.py' so their location is irrelevant.


## Installation and Setup

### Prerequisites

- Python 3.8+
- Raspberry Pi with camera module
- Pybricks-compatible LEGO hub
- Required Python packages (see requirements below)

### Required Python Packages

```bash
# Machine Learning and Computer Vision
pip install torch torchvision 
pip install opencv-python
pip install Pillow

# Camera Interface (Raspberry Pi specific)
pip install picamera2

# Web Dashboard
pip install streamlit
pip install plotly

# Bluetooth Communication
pip install bleak

# File System Monitoring
pip install watchdog
```

### Hardware Setup

1. **Camera**: Connect Raspberry Pi camera module
2. **LEGO Hub Setup**: 
   - Flash the Pybricks firmware to your LEGO Prime Hub
   - Deploy the hub controller MicroPython code (`hub_controller.py`)
   - Connect motors to ports A (pusher 2), B (pusher 1), D (conveyor)
   - Connect ultrasonic sensor to port C (optional)
   - Ensure hub is charged and discoverable via Bluetooth
3. **Physical Components**: 
   - Set up conveyor belt mechanism connected to Motor 3 (Port D)
   - Install pusher mechanisms connected to Motors 1 & 2 (Ports B & A)
   - Align sensors and mechanical components for proper brick detection
4. **Network**: Ensure system has network access for dashboard

### Configuration

#### Setting Parameters
Update the following paths and parameters in the main script:

```python
CLASSIFIER_MODEL_PATH = "/path/to/brick_classifier_simple_torchscript.pt"
JSON_FILE = "/path/to/dashboard_data.json"
SNAPSHOTS_FOLDER = "/path/to/snapshots"
SAME_CLASS_INTERVAL = 6.0  # Seconds between same-class processing
CHECK_INTERVAL = 4.0       # Prediction check frequency
CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence for processing
```

#### Timing Adjustments

Modify pusher delays in `SortingStrategy.get_pusher_delays()` based on your physical setup:

```python
def get_pusher_delays(self, brick_size):
    delays = {
        '2x4': {'pusher1': 2.15, 'pusher2': 3.5},
        # Adjust these values based on your conveyor speed
        # and pusher positioning
    }
    return delays.get(brick_size, {'pusher1': 2.0, 'pusher2': 3.5})
```

## Usage

### Basic Operation

1. **Prepare Hardware**
   - Connect camera to Raspberry Pi
   - Ensure Pybricks hub is powered and discoverable
   - Verify physical sorting mechanisms are operational

2. **Start the System**
   ```bash
   python main.py
   ```

3. **Follow Initialization Sequence**
   - Wait for classifier initialization
   - Allow hub discovery and connection
   - Wait for dashboard startup
   - Press CENTER button on Pybricks hub when prompted
   - Press ENTER when hub light turns cyan

4. **Monitor Operation**
   - Access dashboard at `http://localhost:8501`
   - Observe real-time classifications and metrics
   - Monitor conveyor and pusher operations

## Component Descriptions

### 1. Pybricks Hub Firmware (`hub_controller.py`)

The MicroPython firmware that runs on the LEGO Pybricks Hub, providing the low-level hardware control interface for the physical sorting mechanisms.

**Hardware Configuration:**
- **Motor 1** (Port B): Primary pusher mechanism
- **Motor 2** (Port A): Secondary pusher mechanism  
- **Motor 3** (Port D): Conveyor belt drive motor
- **Ultrasonic Sensor** (Port C): Distance sensing (optional)

**Key Responsibilities:**
- Motor control and coordination
- Bluetooth command reception and processing
- Visual feedback through hub LED colors
- Emergency stop functionality
- Sensor data collection

**Command Protocol:**
The hub accepts single-character commands via Bluetooth and responds with status messages:

| Command | Function | Motor/Action |
|---------|----------|--------------|
| `'1'` | Motor 1 forward rotation (360°) | Primary pusher activation |
| `'2'` | Motor 1 backward rotation (360°) | Primary pusher reverse |
| `'4'` | Motor 2 forward rotation (360°) | Secondary pusher activation |
| `'5'` | Motor 2 backward rotation (360°) | Secondary pusher reverse |
| `'7'` | Motor 3 continuous forward | Conveyor belt forward |
| `'8'` | Motor 3 continuous backward | Conveyor belt reverse |
| `'9'` | Stop Motor 3 | Conveyor belt stop |
| `'0'` | Emergency stop all motors | Safety shutdown |
| `'s'` | Read ultrasonic sensor | Distance measurement |
| `'q'` | Quit/shutdown | Program termination |

**LED Status Indicators:**
- **Cyan**: Ready for commands
- **Orange**: Motor 1 (primary pusher) active
- **Green**: Motor 2 (secondary pusher) active
- **Violet**: Motor 3 (conveyor) active
- **Red**: Emergency stop engaged

**Communication Protocol:**
- Uses non-blocking I/O polling for responsive command processing
- Sends `"rdy\n"` signal to indicate readiness for next command
- Provides command confirmation and status feedback
- Handles communication errors gracefully

**Motor Specifications:**
- **Pusher Motors (1 & 2)**: 400 degrees/second rotation speed
- **Conveyor Motor (3)**: 150 degrees/second continuous operation
- All motors support immediate stop functionality

### 2. BrickClassifier (`brick_classifier.py`)

The core classification component that handles image capture, preprocessing, and inference.

**Key Responsibilities:**
- Camera initialization and control
- Image preprocessing and cropping
- Neural network inference using TorchScript model
- Snapshot management and storage
- Continuous capture with configurable intervals

**Key Methods:**
- `start_continuous_capture()`: Begin real-time classification
- `get_latest_top4()`: Retrieve latest classification results
- `stop_continuous_capture()`: Stop classification process

### 3. SortingOrchestrator (`sorting_orchestrator.py`)

Central coordinator that manages the entire sorting workflow.

**Key Responsibilities:**
- System initialization and component coordination
- Automatic brick sorting workflow
- Queue management for brick processing
- Conveyor and pusher timing control
- Error handling and system shutdown

**Key Methods:**
- `initialize_system()`: Initialize all system components
- `automatic_brick_sorting_loop()`: Main sorting workflow
- `shutdown_system()`: Clean shutdown of all components

### 4. HardwareController (`hardware_controller.py`)

Manages communication with physical hardware via Bluetooth.

**Key Responsibilities:**
- Bluetooth connection to Pybricks hub
- Command transmission to hardware
- Connection status monitoring
- Hardware state tracking

**Supported Commands:**
- `'1'`: Activate pusher 1
- `'4'`: Activate pusher 2
- `'8'`: Start conveyor belt
- `'9'`: Stop conveyor belt

### 5. Dashboard System

#### DashboardConnector (`dashboard_connector.py`)
Manages the Streamlit dashboard as a separate process.

#### Dashboard (`dashboard_v1.py`)
Web-based interface for real-time monitoring and visualization.

**Features:**
- Real-time classification results
- Image gallery of recent snapshots
- Performance metrics and statistics
- Interactive charts and visualizations
- Dark theme interface

### 6. SortingStrategy (`sorting_strategy.py`)

Defines timing and logic for physical sorting operations.

**Key Responsibilities:**
- Pusher timing calculations based on brick size
- Processing duration estimates
- Brick-specific sorting strategies

### 7. UpdateDetector (`update_detector.py`)

Monitors file system changes for dashboard synchronization.

**Key Responsibilities:**
- JSON file monitoring for classification updates
- Image folder monitoring for new snapshots
- Signal-based dashboard refresh mechanism


### Manual Control

The system provides manual control options through the orchestrator:

```python
# Example manual usage
orchestrator = SortingOrchestrator(model_path, json_file, snapshots_folder)
await orchestrator.initialize_system()
await orchestrator.start_classifier_system(interval=5.0, check_interval=1.0)
```

## API Reference

### BrickClassifier API

#### Constructor
```python
BrickClassifier(model_path, snapshot_dir="/path/to/snapshots")
```

#### Methods

##### `start_continuous_capture(same_prediction_interval=2.0, check_interval=1.0)`
Start continuous image capture and classification.

**Parameters:**
- `same_prediction_interval`: Minimum time between saves of same class
- `check_interval`: How often to perform predictions

##### `get_latest_top4()`
Returns the latest top 4 classification results.

**Returns:**
- `tuple`: (classes_list, confidences_list)

##### `stop_continuous_capture()`
Stop continuous capture and cleanup resources.

### HardwareController API

#### Constructor
```python
HardwareController()
```

#### Methods

##### `async find_hub()`
Scan for and locate Pybricks hub.

**Returns:**
- `str`: Hub MAC address if found, None otherwise

##### `async connect_to_hub(address)`
Establish Bluetooth connection to hub.

**Parameters:**
- `address`: MAC address of the hub

**Returns:**
- `bool`: True if connection successful

##### `async send_command(command)`
Send command to connected hub.

**Parameters:**
- `command`: Command string ('1', '4', '8', '9')

**Returns:**
- `bool`: True if command sent successfully

### Dashboard API

#### DashboardConnector

##### `start_dashboard_process()`
Start dashboard and monitoring processes.

##### `update_data(classes, confidences)`
Update dashboard with new classification data.

**Parameters:**
- `classes`: List of class names
- `confidences`: List of confidence scores

##### `stop()`
Stop dashboard and monitoring processes.

## Troubleshooting

### Common Issues

#### 1. Dependency issues on startup
**Symptoms**: Failiure to start program despite requirements being installed

**Solutions**
- Make sure to enable system side dependencies in your venv config
- Confirm no dependency installations failed

#### 2. Camera Initialization Failed
**Symptoms**: Error messages about camera not found

**Solutions**:
- Verify camera module connection
- Check camera is enabled in raspi-config
- Ensure no other processes are using camera

#### 3. Bluetooth Connection Issues
**Symptoms**: Cannot find or connect to Pybricks hub

**Solutions**:
- Ensure hub is powered on and not connected elsewhere
- Check hub name matches `HUB_NAME` in configuration
- Restart Bluetooth service: `sudo systemctl restart bluetooth`

#### 4. Model Loading Errors
**Symptoms**: TorchScript model fails to load

**Solutions**:
- Verify model file path and permissions
- Ensure PyTorch version compatibility
- Check available memory and disk space

#### 5. Pybricks Hub Issues
**Symptoms**: Hub not responding to commands or connection drops

**Solutions**:
- Verify hub firmware is properly flashed with Pybricks
- Check hub battery level and charging status
- Ensure hub MicroPython code is deployed correctly
- Restart hub by pressing center button for 10 seconds
- Verify motor and sensor connections to correct ports
- Check for mechanical obstructions in pusher/conveyor mechanisms

#### 6. Dashboard Not Loading
**Symptoms**: Dashboard doesn't open or shows errors

**Solutions**:
- Check port 8501 is not in use by other processes (all of these should close on startup of main)
- Verify Streamlit installation
- Check file permissions for JSON and snapshot directories

### Performance Optimization

1. **Reduce Image Processing Load**:
   - Adjust camera resolution
   - Optimize crop parameters
   - Alter prediction intervals

2. **Improve Classification Accuracy**:
   - Ensure proper lighting conditions
   - Clean camera lens regularly
   - Calibrate crop regions for optimal brick capture

3. **Hardware Timing**:
   - Fine-tune pusher delays in `SortingStrategy`
   - Adjust conveyor speed settings
   - Monitor for mechanical alignment issues

