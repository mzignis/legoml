import asyncio
import random
from bleak import BleakClient, BleakScanner

# Pybricks Bluetooth UUIDs - using Pybricks-specific service
PYBRICKS_SERVICE_UUID = "c5f50001-8280-46da-89f4-6d8051e4aeef"
PYBRICKS_COMMAND_EVENT_CHAR_UUID = "c5f50002-8280-46da-89f4-6d8051e4aeef"
HUB_NAME = "Pybricks Hub" 

class MLCommandSender:
    def __init__(self):
        self.client = None
        self.connected = False
        self.ready_received = False
        self.available_commands = ['1', '2', '4', '5', '7', '8', '9']
        self.command_count = 0
        
    async def find_hub(self):
        print("Scanning for Pybricks hub...")
        print(f"Looking for hub named: '{HUB_NAME}'")
        
        devices = await BleakScanner.discover(timeout=15.0)  # Longer timeout
        
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
            # print("Waiting for hub ready signal...")
            
            # # Wait for ready signal from hub (if it comes)
            # timeout = 5  # 5 seconds timeout
            # for _ in range(timeout * 10):  # Check 10 times per second
            #     if self.ready_received:
            #         break
            #     await asyncio.sleep(0.1)
            
            # if not self.ready_received:
            #     print("No ready signal received yet - this is normal if hub program isn't running")
                
            self.connected = True
            return True

        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    async def send_command(self, command):
        """Send a command to the hub"""
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
    
    async def continuous_command_loop(self, interval=3.0):
        print(f"Sending random commands every {interval} seconds")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                command = random.choice(self.available_commands)

                success = await self.send_command(command)
                
                if not success:
                    print("Command failed - checking connection...")
                    if not self.client or not self.client.is_connected:
                        print("Lost connection to hub!")
                        break
                    else:
                        print("Command failed but connection seems OK, continuing...")
                
                print(f"Next command in {interval} seconds...")
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nStopping command sender...")
        except Exception as e:
            print(f"Error in command loop: {e}")
    
    async def disconnect(self):
        if self.client and self.connected:
            await self.client.disconnect()
            print("Disconnected from hub")
            self.connected = False

def wait_for_user_input():
    input("Press ENTER after the hub light turns CYAN ")

async def main():
    print("Random Command Sender for Pybricks")
    print("=" * 50)
    
    sender = MLCommandSender()
    
    try:
        hub_address = await sender.find_hub()
        if not hub_address:
            print("\nCould not find hub")
            print("Make sure hub is turned on and is not connected to Pybricks Code")
            return
        
        connected = await sender.connect_to_hub(hub_address)
        if not connected:
            return
        
        print("\nCONNECTION ESTABLISHED!")
        print()
        print("Press the CENTER BUTTON on your Pybricks hub to start the program")
        print("Press ENTER here when the hub light is cyan")
        print()

        wait_for_user_input()
        
        print("\nStarting continuous command transmission...")
        print("Sending random commands every 3 seconds...")
        
        await sender.continuous_command_loop(3.0)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    finally:
        await sender.disconnect()
        print("Program ended")

if __name__ == "__main__":
    asyncio.run(main())