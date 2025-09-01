import asyncio

class SortingStrategy:
    def __init__(self, hardware_controller):
        self.hardware = hardware_controller


    def get_pusher_delays(self, brick_size):
        """Get pusher timing delays for different brick sizes"""
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
        """
        DEPRECATED: This method is kept for backward compatibility but is no longer used
        in the new queueing system. The orchestrator now handles all timing and conveyor management.
        """
        print(f"WARNING: process_brick() called but should not be used with queueing system")
        print(f"Brick info: {brick_type} {brick_size}")


    def get_processing_duration(self, brick_type, brick_size=None):
        """Get the total processing duration for a brick type"""
        if brick_type == "damaged":
            delays = self.get_pusher_delays(brick_size)
            # Total time is max pusher delay + small buffer for pusher action
            return max(delays['pusher1'], delays['pusher2']) + 0.5
        elif brick_type == "undamaged":
            # Undamaged bricks just pass through
            return 3.0
        else:
            return 3.0  # Default fallback


    async def fire_pusher(self, pusher_number, brick_info):
        """Fire a specific pusher for a brick"""
        size_display = brick_info['size'] if brick_info['size'] else "unknown size"
        brick_type = brick_info['type']
        
        if pusher_number == 1:
            print(f"Activating front pusher for {brick_type} {size_display} brick")
            await self.hardware.send_command('1')
        elif pusher_number == 2:
            print(f"Activating second pusher for {brick_type} {size_display} brick")
            await self.hardware.send_command('4')
        else:
            print(f"Unknown pusher number: {pusher_number}")


    def calculate_pusher_times(self, brick_info, detection_time):
        """Calculate when pushers should fire for a given brick"""
        brick_type = brick_info['type']
        brick_size = brick_info['size']
        
        if brick_type == "damaged":
            delays = self.get_pusher_delays(brick_size)
            return {
                'pusher1_time': detection_time + delays['pusher1'],
                'pusher2_time': detection_time + delays['pusher2']
            }
        else:
            # Undamaged bricks don't need pushers
            return {}


    def get_brick_description(self, brick_type, brick_size=None):
        """Get a human-readable description of a brick"""
        size_display = brick_size if brick_size else "unknown size"
        return f"{brick_type} {size_display} brick"