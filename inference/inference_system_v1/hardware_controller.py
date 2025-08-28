from bleak import BleakClient, BleakScanner

class HardwareController:
    def __init__(self):
        self.client = None
        self.connected = False
        self.ready_received = False
        self.command_count = 0
        self.conveyor_running = False
        
        # Pybricks Bluetooth UUIDs
        self.PYBRICKS_SERVICE_UUID = "c5f50001-8280-46da-89f4-6d8051e4aeef"
        self.PYBRICKS_COMMAND_EVENT_CHAR_UUID = "c5f50002-8280-46da-89f4-6d8051e4aeef"
        self.HUB_NAME = "Pybricks Hub"


    async def find_hub(self):
        print("Scanning for Pybricks hub...")
        print(f"Looking for hub named: '{self.HUB_NAME}'")
        
        devices = await BleakScanner.discover(timeout=15.0)  # Longer timeout
        
        # we need to do this as the hub uses a random non-persistent MAC address
        for device in devices:
            if device.name and self.HUB_NAME in device.name:
                print(f"Found hub: {device.name} ({device.address})")
                return device.address
        
        print(f"Hub '{self.HUB_NAME}' not found!")
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
                    if service.uuid.lower() == self.PYBRICKS_SERVICE_UUID.lower():
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
            
            await self.client.start_notify(self.PYBRICKS_COMMAND_EVENT_CHAR_UUID, handle_response)
            
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
                self.PYBRICKS_COMMAND_EVENT_CHAR_UUID, 
                full_command,
                response=True
            )
            
            self.command_count += 1
            print(f"Sent command '{command}' (#{self.command_count})")
            
            # Update conveyor state tracking
            if command == '8':
                self.conveyor_running = True
            elif command == '9':
                self.conveyor_running = False
                
            return True
            
        except Exception as e:
            print(f"Failed to send command '{command}': {e}")
            return False


    async def disconnect(self):
        if self.client and self.connected:
            await self.client.disconnect()
            print("Disconnected from hub")
            self.connected = False