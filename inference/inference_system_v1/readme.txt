Class Breakdown:

main.py                      # Clean main program
├── hardware_controller.py   # Bluetooth communication
├── classifier_manager.py    # AI classification handling  
├── sorting_strategy.py      # Brick processing logic
└── sorting_orchestrator.py  # System coordination

HardwareController:
├── send_command('8') → Start conveyor
├── send_command('9') → Stop conveyor  
├── send_command('1') → Activate front pusher
└── send_command('4') → Activate second pusher

SortingStrategy:
├── get_pusher_delays() → Calculate timing
├── process_brick() → Coordinate sequence
└── delayed_pusher_activation() → Timed actions

ClassifierManager:
├── get_latest_predictions() → AI results
└── parse_classification() → Format conversion

SortingOrchestrator:
├── initialize_system() → Setup all components
├── automatic_brick_sorting_loop() → Main coordination
└── shutdown_system() → Clean shutdown



Command Flow: 

Main Loop → Orchestrator → Strategy → Hardware
                    ↓
                Classifier



Program Summary:

main()
├── SortingOrchestrator.__init__()
│   ├── HardwareController.__init__()
│   │   └── Initialize variables (client=None, connected=False, etc.)
│   ├── ClassifierManager.__init__()
│   │   └── Store classifier_model_path
│   └── SortingStrategy.__init__()
│       └── Store hardware_controller reference
├── orchestrator.initialize_system()
│   ├── classifier.initialize_classifier()
│   │   ├── Create BrickClassifier instance
│   │   ├── Load trained model from disk
│   │   └── Set classifier ready for use
│   ├── hardware.find_hub()
│   │   ├── BleakScanner.discover() [scan for 15 seconds]
│   │   ├── Filter devices by name "Pybricks Hub"
│   │   └── Return hub Bluetooth address or None
│   └── hardware.connect_to_hub()
│       ├── Create BleakClient with hub address
│       ├── client.connect() [establish Bluetooth connection]
│       ├── client.get_services() [verify Pybricks service available]
│       ├── Set up notification handler for hub responses
│       │   └── handle_response() [processes "rdy" and other hub messages]
│       ├── client.start_notify() [listen for hub responses]
│       └── Set connected=True
├── wait_for_user_input()
│   └── input() [wait for user to press ENTER after hub setup]
├── orchestrator.start_classifier_system()
│   └── classifier.start_classifier()
│       └── classifier.start_continuous_capture() [start camera at 3s intervals]
└── orchestrator.automatic_brick_sorting_loop()
    └── Main loop (every 5 seconds):
        ├── Check hardware.conveyor_running
        │   └── Skip iteration if busy
        ├── classifier.get_latest_predictions()
        │   ├── classifier.get_latest_top4()
        │   │   ├── Get AI predictions from camera
        │   │   └── Return top 4 classes + confidence scores
        │   └── Handle exceptions and return results
        ├── Filter predictions (confidence > 0.5 + new prediction)
        ├── classifier.parse_classification()
        │   ├── Split "color_shape_state" format
        │   ├── Map state to brick_type (damaged/undamaged)
        │   ├── Extract brick_size from shape
        │   └── Return (brick_type, brick_size, color, state)
        ├── strategy.process_brick()
        │   ├── IF damaged brick:
        │   │   ├── strategy.get_pusher_delays()
        │   │   │   ├── Look up size-specific timings
        │   │   │   │   ├── 2x4: pusher1=2.15s, pusher2=3.5s
        │   │   │   │   ├── 2x2: pusher1=2.0s, pusher2=3.5s
        │   │   │   │   ├── 1x3: pusher1=1.8s, pusher2=3.5s
        │   │   │   │   └── etc.
        │   │   │   └── Return delay dictionary
        │   │   ├── hardware.send_command('8') [start conveyor belt]
        │   │   │   ├── Encode command as UTF-8 bytes
        │   │   │   ├── Prepend 0x06 byte (Pybricks protocol)
        │   │   │   ├── client.write_gatt_char() [send via Bluetooth]
        │   │   │   └── Set hardware.conveyor_running = True
        │   │   ├── Create concurrent pusher tasks:
        │   │   │   ├── strategy.delayed_pusher_activation('1', delay1, "Front pusher")
        │   │   │   │   ├── asyncio.sleep(delay) [wait for brick position]
        │   │   │   │   └── hardware.send_command('1') [activate front pusher]
        │   │   │   └── strategy.delayed_pusher_activation('4', delay2, "Second pusher")
        │   │   │       ├── asyncio.sleep(delay) [wait for brick position]
        │   │   │       └── hardware.send_command('4') [activate second pusher]
        │   │   ├── asyncio.gather() [wait for both pushers to complete]
        │   │   ├── Additional timing buffer
        │   │   └── hardware.send_command('9') [stop conveyor belt]
        │   │       └── Set hardware.conveyor_running = False
        │   └── IF undamaged brick:
        │       ├── hardware.send_command('8') [start conveyor belt]
        │       │   └── Set hardware.conveyor_running = True
        │       ├── asyncio.sleep(3.0) [let brick pass through]
        │       └── hardware.send_command('9') [stop conveyor belt]
        │           └── Set hardware.conveyor_running = False
        ├── Update last_processed_prediction [prevent reprocessing]
        ├── asyncio.sleep(check_interval) [wait 5 seconds]
        └── Handle KeyboardInterrupt (Ctrl+C):
            ├── Set running=False
            ├── Emergency stop conveyor if needed
            └── Break from loop

Finally:
└── orchestrator.shutdown_system()
    ├── hardware.disconnect()
    │   └── client.disconnect() [close Bluetooth connection]
    └── classifier.cleanup()
        └── classifier.cleanup() [stop camera, free resources]
