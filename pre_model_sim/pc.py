import asyncio
import random
from bleak import BleakClient, BleakScanner

# Pybricks Bluetooth UUIDs - using Pybricks-specific service
PYBRICKS_SERVICE_UUID = "c5f50001-8280-46da-89f4-6d8051e4aeef"
PYBRICKS_COMMAND_EVENT_CHAR_UUID = "c5f50002-8280-46da-89f4-6d8051e4aeef"
HUB_NAME = "Pybricks Hub" 

class BrickSortingController:
    def __init__(self):
        self.client = None
        self.connected = False
        self.ready_received = False
        self.command_count = 0
        self.conveyor_running = False
        
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
            
            # Set up notifications to receive responses from hub
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
        """Get pusher delays for specific brick sizes"""
        delays = {
            '2x4': {'pusher1': 2.15, 'pusher2': 3.5},  # 2.1-2.2 average
            '2x2': {'pusher1': 2.0, 'pusher2': 3.5},
            '1x3': {'pusher1': 1.8, 'pusher2': 3.5},
            '1x6': {'pusher1': 2.4, 'pusher2': 3.5},
            '2x6': {'pusher1': 2.4, 'pusher2': 3.5}
        }
        return delays.get(brick_size, {'pusher1': 2.0, 'pusher2': 3.5})  # Default fallback
    
    async def process_brick(self, brick_type, brick_size=None):
        """Process a brick based on its type (damaged or undamaged) and size"""
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
            
            print(f"✓ Damaged {brick_size} brick successfully removed from line!")
            
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
            
            print("✓ Undamaged brick passed through successfully!")
        
        print("--- Brick processing complete ---\n")
    
    async def process_brick_default_damaged(self):
        """Process damaged brick with default timing"""
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
        
        print("✓ Damaged brick successfully removed from line!")
    
    async def delayed_pusher_activation(self, command, delay, pusher_name):
        """Activate a pusher after a specified delay"""
        await asyncio.sleep(delay)
        print(f"Activating {pusher_name} (Command {command}) after {delay}s delay")
        await self.send_command(command)
    
    def parse_command_input(self, user_input):
        """Parse CLI-style command input"""
        parts = user_input.strip().split()
        
        if len(parts) == 0:
            return None, None, "Empty input"
        
        # Handle single character commands for backward compatibility
        if len(parts) == 1:
            cmd = parts[0].lower()
            if cmd in ['u', 'undamaged']:
                return 'undamaged', None, None
            elif cmd in ['d', 'damaged']:
                return 'damaged', None, None
            elif cmd in ['q', 'quit', 'exit']:
                return 'quit', None, None
            else:
                return None, None, f"Unknown command: {cmd}"
        
        # Handle CLI-style commands: -d -2x4, -u, etc.
        if len(parts) >= 1:
            first_part = parts[0].lower()
            
            if first_part == '-u' or first_part == '-undamaged':
                return 'undamaged', None, None
            
            elif first_part == '-d' or first_part == '-damaged':
                if len(parts) == 1:
                    return 'damaged', None, None  # No size specified
                elif len(parts) == 2:
                    size_arg = parts[1].lower()
                    if size_arg.startswith('-'):
                        size_arg = size_arg[1:]  # Remove leading dash
                    
                    valid_sizes = ['2x4', '2x2', '1x3', '1x6', '2x6']
                    if size_arg in valid_sizes:
                        return 'damaged', size_arg, None
                    else:
                        return None, None, f"Invalid brick size: {size_arg}. Valid sizes: {', '.join(valid_sizes)}"
                else:
                    return None, None, "Too many arguments for damaged brick command"
            
            elif first_part == '-q' or first_part == '-quit':
                return 'quit', None, None
            
            else:
                return None, None, f"Unknown command: {first_part}"
        
        return None, None, "Could not parse command"
    
    async def brick_sorting_loop(self):
        print("Brick Sorting Simulation Active")
        print("=" * 50)
        print("Commands:")
        print("  Simple format:")
        print("    u, undamaged     - Process undamaged brick")
        print("    d, damaged       - Process damaged brick (default timing)")
        print("    q, quit          - Exit program")
        print()
        print("  CLI format:")
        print("    -u               - Process undamaged brick")
        print("    -d               - Process damaged brick (default timing)")
        print("    -d -2x4          - Process damaged 2x4 brick")
        print("    -d -2x2          - Process damaged 2x2 brick")
        print("    -d -1x3          - Process damaged 1x3 brick")
        print("    -d -1x6          - Process damaged 1x6 brick")
        print("    -d -2x6          - Process damaged 2x6 brick")
        print("    -q               - Exit program")
        print()
        print("Brick size timing:")
        print("  2x4: Pusher1@2.15s, Pusher2@3.5s")
        print("  2x2: Pusher1@2.0s,  Pusher2@3.5s")
        print("  1x3: Pusher1@1.8s,  Pusher2@3.5s")
        print("  1x6: Pusher1@2.4s,  Pusher2@3.5s")
        print("  2x6: Pusher1@2.4s,  Pusher2@3.5s")
        print()
        
        try:
            while True:
                if self.conveyor_running:
                    print("⚠️  Conveyor is running - please wait...")
                    await asyncio.sleep(0.5)
                    continue
                
                user_input = input("Enter command: ").strip()
                
                if not user_input:
                    continue
                
                brick_type, brick_size, error = self.parse_command_input(user_input)
                
                if error:
                    print(f"❌ Error: {error}")
                    continue
                
                if brick_type == 'quit':
                    print("Shutting down brick sorting system...")
                    break
                elif brick_type == 'undamaged':
                    await self.process_brick("undamaged")
                elif brick_type == 'damaged':
                    await self.process_brick("damaged", brick_size)
                
                # Small delay before next brick
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        except Exception as e:
            print(f"Error in sorting loop: {e}")
        
        # Emergency stop if conveyor is still running
        if self.conveyor_running:
            print("Emergency stop - stopping conveyor...")
            await self.send_command('9')
            self.conveyor_running = False
    
    async def disconnect(self):
        if self.client and self.connected:
            await self.client.disconnect()
            print("Disconnected from hub")
            self.connected = False

def wait_for_user_input():
    input("Press ENTER after the hub light turns CYAN ")

async def main():
    print("Brick Sorting Controller for Pybricks")
    print("=" * 50)
    
    controller = BrickSortingController()
    
    try:
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
        
        print("\nStarting brick sorting simulation...")
        print("You can now input brick types to simulate ML model predictions")
        
        await controller.brick_sorting_loop()
        
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    finally:
        await controller.disconnect()
        print("Program ended")

if __name__ == "__main__":
    asyncio.run(main())