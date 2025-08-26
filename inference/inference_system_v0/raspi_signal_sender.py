import asyncio
from bleak import BleakClient, BleakScanner
from brick_classifier import BrickClassifier
import time

# Pybricks Bluetooth UUIDs - using Pybricks-specific service
PYBRICKS_SERVICE_UUID = "c5f50001-8280-46da-89f4-6d8051e4aeef"
PYBRICKS_COMMAND_EVENT_CHAR_UUID = "c5f50002-8280-46da-89f4-6d8051e4aeef"
HUB_NAME = "Pybricks Hub" 

class BrickSortingController:
    def __init__(self, classifier_model_path="/home/candfpi4b/lego_pdm/brick_classifier_simple96.pth"):
        self.client = None
        self.connected = False
        self.ready_received = False
        self.command_count = 0
        self.conveyor_running = False
        self.classifier = None
        self.classifier_model_path = classifier_model_path
        self.running = False
        

    async def find_hub(self):
        print("Scanning for Pybricks hub...")
        print(f"Looking for hub named: '{HUB_NAME}'")
        
        devices = await BleakScanner.discover(timeout=15.0)  # Longer timeout
        
        # we need to do this as the hub uses a random non-persistent MAC address
        for device in devices:
            if device.name and HUB_NAME in device.name:
                print(f"Found hub: {device.name} ({device.address})")
                return device.address
        
        print(f"Hub '{HUB_NAME}' not found!")
        print("Available devices:")
        for device in devices:
            if device.name:
                print(f"  - {device.name} ({device.address})")
        
        return None
    

    async def connect_to_hub(self, address):
        try:
            print(f"Connecting to hub at {address}...")
            self.client = BleakClient(address)
            await self.client.connect()
            
            # Verify connection by checking if we can access the Pybricks service
            try:
                # Check if Pybricks service is available
                services = await self.client.get_services()
                pybricks_service_found = False
                for service in services:
                    if service.uuid.lower() == PYBRICKS_SERVICE_UUID.lower():
                        pybricks_service_found = True
                        print("Pybricks service confirmed")
                        break
                
                if not pybricks_service_found:
                    print("Warning: Pybricks service not found")
                    print("Available services:")
                    for service in services:
                        print(f"  - {service.uuid}")
                
            except Exception as e:
                print(f"Warning: Could not verify Pybricks service: {e}")
                print("Continuing anyway...")
            
            # notifications to receive responses from hub
            def handle_response(sender, data: bytearray):
                try:
                    if len(data) > 0 and data[0] == 0x01:  # "write stdout" event (0x01)
                        payload = data[1:]  # Remove the event byte
                        response = payload.decode('utf-8').strip()
                        if response == "rdy":
                            print("Hub is ready!")
                            self.ready_received = True
                        elif response:
                            print(f"Hub response: {response}")
                except Exception as e:
                    print(f"Response decode error: {e}")
            
            await self.client.start_notify(PYBRICKS_COMMAND_EVENT_CHAR_UUID, handle_response)
            
            print("Connected to Pybricks hub!")
            self.connected = True
            return True

        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    

    def initialize_classifier(self):
        try:
            print("Initializing brick classifier...")
            self.classifier = BrickClassifier(self.classifier_model_path)
            print("Brick classifier initialized successfully!")
            return True
        except Exception as e:
            print(f"Failed to initialize classifier: {e}")
            return False
    

    def start_classifier(self, prediction_interval=3.0):
        try:
            print(f"Starting continuous capture with {prediction_interval}s intervals...")
            self.classifier.start_continuous_capture(prediction_interval=prediction_interval)
            print("Classifier capture started!")
            return True
        except Exception as e:
            print(f"Failed to start classifier: {e}")
            return False
    

    def parse_classification(self, classification):
        """
        Parse classification string format: {color}_{shape}_{state}
        Returns: (brick_type, brick_size, color, state)
        where brick_type is 'damaged' or 'undamaged'
        """
        if not classification:
            return None, None, None, None
        
        try:
            parts = classification.split('_')
            if len(parts) != 3:
                print(f"Warning: Unexpected classification format: {classification}")
                return None, None, None, None
            
            color, shape, state = parts
            
            # Map state to our brick_type
            if state.lower() in ['good', 'undamaged']:
                brick_type = 'undamaged'
            else:
                brick_type = 'damaged'
            
            # Use shape as brick_size
            brick_size = shape.lower()
            
            return brick_type, brick_size, color, state
        
        except Exception as e:
            print(f"Error parsing classification '{classification}': {e}")
            return None, None, None, None
    

    async def send_command(self, command):
        if not self.connected or not self.client:
            print("Not connected to hub!")
            return False
        
        try:
            # Prepare command as bytes with newline
            command_data = f"{command}\n".encode('utf-8')
            
            # Send command via Pybricks command/event characteristic
            # Prepend with 0x06 byte (write stdin command)
            full_command = b"\x06" + command_data
            
            await self.client.write_gatt_char(
                PYBRICKS_COMMAND_EVENT_CHAR_UUID, 
                full_command,
                response=True
            )
            
            self.command_count += 1
            print(f"Sent command '{command}' (#{self.command_count})")
            return True
            
        except Exception as e:
            print(f"Failed to send command '{command}': {e}")
            return False
    

    def get_pusher_delays(self, brick_size):
        delays = {
            '2x4': {'pusher1': 2.15, 'pusher2': 3.5},
            '2x2': {'pusher1': 2.0, 'pusher2': 3.5},
            '1x3': {'pusher1': 1.8, 'pusher2': 3.5},
            '1x6': {'pusher1': 2.4, 'pusher2': 3.5},
            '2x6': {'pusher1': 2.4, 'pusher2': 3.5}
        }
        return delays.get(brick_size, {'pusher1': 2.0, 'pusher2': 3.5})  # Default fallback
    

    async def process_brick(self, brick_type, brick_size=None):
        if brick_type == "damaged" and brick_size:
            print(f"\n--- Processing damaged {brick_size} brick ---")
            
            # Get delays for this brick size
            delays = self.get_pusher_delays(brick_size)
            pusher1_delay = delays['pusher1']
            pusher2_delay = delays['pusher2']
            
            print(f"Brick size: {brick_size}")
            print(f"Pusher 1 delay: {pusher1_delay}s")
            print(f"Pusher 2 delay: {pusher2_delay}s")
            print("Activating pushers...")
            
            # Start conveyor belt
            print("Starting conveyor belt...")
            await self.send_command('8')
            self.conveyor_running = True
            
            # Create tasks for delayed pusher activation with size-specific delays
            pusher1_task = asyncio.create_task(self.delayed_pusher_activation('1', pusher1_delay, f"Front pusher ({brick_size})"))
            pusher2_task = asyncio.create_task(self.delayed_pusher_activation('4', pusher2_delay, f"Second pusher ({brick_size})"))
            
            # Calculate total runtime (max delay + small buffer for pusher action)
            total_runtime = max(pusher1_delay, pusher2_delay) + 0.5
            
            # Wait for all operations to complete
            await asyncio.gather(pusher1_task, pusher2_task)
            
            # Ensure minimum runtime before stopping conveyor
            await asyncio.sleep(max(0, total_runtime - max(pusher1_delay, pusher2_delay)))
            
            print("Stopping conveyor belt...")
            await self.send_command('9')
            self.conveyor_running = False
            
            print(f"Damaged {brick_size} brick successfully removed from line!")
        
        elif brick_type == "damaged" and not brick_size:
            print("\n--- Processing damaged brick (unknown size) ---")
            print("Using default timing...")
            
            # Use default delays if no size specified
            await self.process_brick_default_damaged()

        elif brick_type == "undamaged":
            print("\n--- Processing undamaged brick ---")
            print("Undamaged brick - letting it pass through...")
            
            # Start conveyor belt
            print("Starting conveyor belt...")
            await self.send_command('8')
            self.conveyor_running = True
            
            # Wait 3 seconds for brick to pass through
            await asyncio.sleep(3.0)
            
            print("Stopping conveyor belt...")
            await self.send_command('9')
            self.conveyor_running = False
            
            print("Undamaged brick passed through successfully!")
        
        print("--- Brick processing complete ---\n")
    

    async def process_brick_default_damaged(self):
        # Start conveyor belt
        print("Starting conveyor belt...")
        await self.send_command('8')
        self.conveyor_running = True
        
        # Create tasks for delayed pusher activation with original delays
        pusher1_task = asyncio.create_task(self.delayed_pusher_activation('1', 0.4, "Front pusher (default)"))
        pusher2_task = asyncio.create_task(self.delayed_pusher_activation('4', 0.7, "Second pusher (default)"))
        
        # Wait for pushers to complete (they run concurrently)
        await asyncio.gather(pusher1_task, pusher2_task)
        
        # Wait total of 3 seconds from conveyor start, then stop
        await asyncio.sleep(3.0 - max(0.4, 0.7))  # Subtract the longest pusher delay
        
        print("Stopping conveyor belt...")
        await self.send_command('9')
        self.conveyor_running = False
        
        print("Damaged brick successfully removed from line!")
    

    async def delayed_pusher_activation(self, command, delay, pusher_name):
        await asyncio.sleep(delay)
        print(f"Activating {pusher_name} (Command {command}) after {delay}s delay")
        await self.send_command(command)
    

    async def automatic_brick_sorting_loop(self, confidence_threshold=0.5, check_interval=1.0):
        print("AUTOMATIC BRICK SORTING ACTIVE")
        print("=" * 50)
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Check interval: {check_interval}s")
        print("The system will automatically detect and sort bricks!")
        print("Press Ctrl+C to stop the system")
        print()
        
        self.running = True
        last_processed_prediction = None
        
        try:
            while self.running:
                # Skip if conveyor is currently running
                if self.conveyor_running:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get latest predictions from classifier
                try:
                    classes, confidences = self.classifier.get_latest_top4()
                    
                    if classes and confidences:
                        top_prediction = classes[0]
                        top_confidence = confidences[0]
                        
                        # Only process if confidence is above threshold and it's a new prediction
                        if top_confidence >= confidence_threshold and top_prediction != last_processed_prediction:
                            print(f"New detection: {top_prediction} (confidence: {top_confidence:.3f})")
                            
                            # Parse the classification
                            brick_type, brick_size, color, state = self.parse_classification(top_prediction)
                            
                            if brick_type:
                                print(f"Parsed: Type={brick_type}, Size={brick_size}, Color={color}, State={state}")
                                
                                # Process the brick
                                await self.process_brick(brick_type, brick_size)
                                
                                # Update last processed to avoid reprocessing same prediction
                                last_processed_prediction = top_prediction
                                
                            else:
                                print(f"Could not parse classification: {top_prediction}")
                        
                        elif top_confidence < confidence_threshold:
                            print(f"Waiting... Current prediction: {top_prediction} (confidence: {top_confidence:.3f} < {confidence_threshold})")
                    
                    else:
                        print("Waiting for classifier predictions...")
                
                except Exception as e:
                    print(f"Error getting classifier predictions: {e}")
                
                # Wait before next check
                await asyncio.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nShutdown requested by user...")
            self.running = False

        except Exception as e:
            print(f"Error in automatic sorting loop: {e}")
            self.running = False
        
        # Emergency stop if conveyor is still running
        if self.conveyor_running:
            print("mergency stop - stopping conveyor...")
            await self.send_command('9')
            self.conveyor_running = False
        
        print("Automatic sorting stopped.")
    

    async def disconnect(self):
        """Disconnect from hub and cleanup classifier"""
        if self.client and self.connected:
            await self.client.disconnect()
            print("Disconnected from hub")
            self.connected = False
        
        if self.classifier:
            try:
                self.classifier.cleanup()
                print("Classifier cleaned up")
            except Exception as e:
                print(f"Error cleaning up classifier: {e}")


def wait_for_user_input():
    input("Press ENTER after the hub light turns CYAN ")


async def main():
    print("AUTOMATIC BRICK SORTING CONTROLLER")
    print("=" * 50)
    
    # You can customize these parameters
    CLASSIFIER_MODEL_PATH = "/home/candfpi4b/lego_pdm/brick_classifier_simple96.pth"
    PREDICTION_INTERVAL = 3.0  # How often classifier makes predictions (seconds)
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to process a brick
    CHECK_INTERVAL = 5.0  # How often to check for new predictions (seconds)
    
    controller = BrickSortingController(CLASSIFIER_MODEL_PATH)
    
    try:
        # Step 1: Initialize classifier
        print("Step 1: Initializing brick classifier...")
        if not controller.initialize_classifier():
            print("Failed to initialize classifier. Exiting.")
            return
        
        # Step 2: Find and connect to hub
        print("\nStep 2: Connecting to Pybricks hub...")
        hub_address = await controller.find_hub()
        if not hub_address:
            print("\nCould not find hub")
            print("Make sure hub is turned on and is not connected to Pybricks Code")
            return
        
        connected = await controller.connect_to_hub(hub_address)
        if not connected:
            return
        
        print("\nCONNECTION ESTABLISHED!")
        print()
        print("Press the CENTER BUTTON on your Pybricks hub to start the program")
        print("Press ENTER here when the hub light is cyan")
        print()

        wait_for_user_input()
        
        # Step 3: Start classifier
        print("\nStep 3: Starting brick classifier...")
        if not controller.start_classifier(PREDICTION_INTERVAL):
            print("Failed to start classifier. Exiting.")
            return
        
        print("\nSYSTEM READY!")
        print("The system will now automatically detect and sort bricks!")
        print(f"Prediction interval: {PREDICTION_INTERVAL}s")
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"Check interval: {CHECK_INTERVAL}s")
        print()
        
        # Step 4: Start automatic sorting
        await controller.automatic_brick_sorting_loop(
            confidence_threshold=CONFIDENCE_THRESHOLD,
            check_interval=CHECK_INTERVAL
        )
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    finally:
        await controller.disconnect()
        print("🔚 Program ended")


if __name__ == "__main__":
    asyncio.run(main())