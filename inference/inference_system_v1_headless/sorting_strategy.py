import asyncio

class SortingStrategy:
    def __init__(self, hardware_controller):
        self.hardware = hardware_controller


    def get_pusher_delays(self, brick_size):
        delays = {
            '2x4': {'pusher1': 2.15, 'pusher2': 3.5},
            '2x2': {'pusher1': 2.0, 'pusher2': 3.5},
            '1x3': {'pusher1': 1.8, 'pusher2': 3.5},
            '1x6': {'pusher1': 2.4, 'pusher2': 3.5},
            '2x6': {'pusher1': 2.4, 'pusher2': 3.5}
        }
        # Return defaults for unknown sizes or None
        return delays.get(brick_size, {'pusher1': 2.0, 'pusher2': 3.5})


    async def process_brick(self, brick_type, brick_size=None):
        if brick_type == "damaged":
            # Handle both sized and unsized damaged bricks
            size_display = brick_size if brick_size else "unknown size"
            print(f"\n--- Processing damaged {size_display} brick ---")
            
            # Get delays for this brick size (will use defaults if size is None)
            delays = self.get_pusher_delays(brick_size)
            pusher1_delay = delays['pusher1']
            pusher2_delay = delays['pusher2']
            
            print(f"Brick size: {size_display}")
            print(f"Pusher 1 delay: {pusher1_delay}s")
            print(f"Pusher 2 delay: {pusher2_delay}s")
            print("Activating pushers...")
            
            # Start conveyor belt
            print("Starting conveyor belt...")
            await self.hardware.send_command('8')
            
            # Create tasks for delayed pusher activation
            pusher1_task = asyncio.create_task(self.delayed_pusher_activation('1', pusher1_delay, f"Front pusher ({size_display})"))
            pusher2_task = asyncio.create_task(self.delayed_pusher_activation('4', pusher2_delay, f"Second pusher ({size_display})"))
            
            # Calculate total runtime (max delay + small buffer for pusher action)
            total_runtime = max(pusher1_delay, pusher2_delay) + 0.5
            
            # Wait for all operations to complete
            await asyncio.gather(pusher1_task, pusher2_task)
            
            # Ensure minimum runtime before stopping conveyor
            await asyncio.sleep(max(0, total_runtime - max(pusher1_delay, pusher2_delay)))
            
            print("Stopping conveyor belt...")
            await self.hardware.send_command('9')
            
            print(f"Damaged {size_display} brick successfully removed from line!")

        elif brick_type == "undamaged":
            print("\n--- Processing undamaged brick ---")
            print("Undamaged brick - letting it pass through...")
            
            # Start conveyor belt
            print("Starting conveyor belt...")
            await self.hardware.send_command('8')
            
            # Wait 3 seconds for brick to pass through
            await asyncio.sleep(3.0)
            
            print("Stopping conveyor belt...")
            await self.hardware.send_command('9')
            
            print("Undamaged brick passed through successfully!")
        
        print("--- Brick processing complete ---\n")


    async def delayed_pusher_activation(self, command, delay, pusher_name):
        await asyncio.sleep(delay)
        print(f"Activating {pusher_name} (Command {command}) after {delay}s delay")
        await self.hardware.send_command(command)