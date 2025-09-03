import asyncio
import time
import subprocess
import sys
from hardware_controller import HardwareController
from classifier_manager import ClassifierManager
from sorting_strategy import SortingStrategy
from dashboard_connector import DashboardConnector


class SortingOrchestrator:
    def __init__(self, classifier_model_path):
        self.hardware = HardwareController()
        self.classifier = ClassifierManager(classifier_model_path)
        self.strategy = SortingStrategy(self.hardware)
        self.dashboard = DashboardConnector()
        self.running = False
        self.brick_queue = []  # Queue of pending brick processing tasks
        self.conveyor_stop_time = 0  # When the conveyor should stop
        self.conveyor_task = None  # Task managing conveyor operations

    def cleanup_port_8501(self):
        """Kill any process using port 8501 before starting dashboard"""
        try:
            # Find process using port 8501
            result = subprocess.run(['lsof', '-i', ':8501'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                
                # Skip header line, process each process line
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        process_name = parts[0]
                        print(f"Found process {process_name} (PID: {pid}) using port 8501")
                        
                        try:
                            # Kill the process
                            subprocess.run(['kill', '-9', pid], check=True, timeout=5)
                            print(f"Successfully killed process {pid}")
                        except subprocess.CalledProcessError as e:
                            print(f"Failed to kill process {pid}: {e}")
                        except subprocess.TimeoutExpired:
                            print(f"Timeout while trying to kill process {pid}")
            else:
                print("No process found using port 8501")
                
        except subprocess.TimeoutExpired:
            print("Timeout while checking for processes on port 8501")
        except FileNotFoundError:
            print("lsof command not found. Skipping port cleanup.")
        except Exception as e:
            print(f"Error during port cleanup: {e}")

    async def initialize_system(self):
        # Step 1: Initialize classifier
        print("Step 1: Initializing brick classifier...")
        if not self.classifier.initialize_classifier():
            print("Failed to initialize classifier. Exiting.")
            return False
        
        # Step 2: Find and connect to hub
        print("\nStep 2: Connecting to Pybricks hub...")
        hub_address = await self.hardware.find_hub()
        if not hub_address:
            print("\nCould not find hub")
            print("Make sure hub is turned on and is not connected to Pybricks Code")
            return False
        
        connected = await self.hardware.connect_to_hub(hub_address)
        if not connected:
            return False
        
        print("\nCONNECTION ESTABLISHED!")

        # Step 3: Clean up port 8501 before starting dashboard
        print("\nStep 3: Cleaning up port 8501...")
        self.cleanup_port_8501()
        
        # Step 4: Start dashboard in separate process
        print("\nStep 4: Starting dashboard...")
        self.dashboard.start_dashboard_process()
        return True

    async def start_classifier_system(self, same_prediction_interval, check_interval):
        print("\nStep 5: Starting brick classifier...")
        if not self.classifier.start_classifier(same_prediction_interval, check_interval):
            print("Failed to start classifier. Exiting.")
            return False
        return True

    async def automatic_brick_sorting_loop(self, confidence_threshold=0.5, check_interval=1.0, min_processing_interval=None):
        # Use check_interval as min_processing_interval if not specified
        if min_processing_interval is None:
            min_processing_interval = check_interval
            
        print("AUTOMATIC BRICK SORTING ACTIVE")
        print("=" * 50)
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Check interval: {check_interval}s")
        print(f"Min processing interval for same class: {min_processing_interval}s")
        print("The system will automatically detect and sort bricks!")
        print("Press Ctrl+C to stop the system")
        print()
        
        self.running = True
        last_processed_class = None
        last_processing_time = 0
        
        # Start the conveyor management task
        self.conveyor_task = asyncio.create_task(self.manage_conveyor_and_pushers())
        
        try:
            while self.running:
                # Get latest predictions from classifier (NO LONGER BLOCKED BY CONVEYOR STATE)
                try:
                    classes, confidences = self.classifier.get_latest_predictions()
                    
                    if classes and confidences:
                        # Non-blocking dashboard update
                        self.dashboard.update_data(classes, confidences)

                        top_prediction = classes[0]
                        top_confidence = confidences[0]
                        
                        # Check if we should process this prediction
                        should_process = False
                        current_time = time.time()
                        
                        if top_confidence >= confidence_threshold:
                            # Always process if class changed or it's the first prediction
                            if last_processed_class is None or top_prediction != last_processed_class:
                                should_process = True
                                reason = "new class detected" if last_processed_class else "first detection"
                            # For same class: only process if enough time has passed
                            elif current_time - last_processing_time >= min_processing_interval:
                                should_process = True
                                reason = f"same class after {min_processing_interval}s interval"
                            else:
                                time_remaining = min_processing_interval - (current_time - last_processing_time)
                                print(f"Same class '{top_prediction}' - waiting {time_remaining:.1f}s more before processing")
                        
                        if should_process:
                            print(f"Queueing brick: {top_prediction} (confidence: {top_confidence:.3f}) - {reason}")
                            
                            # Parse the classification
                            brick_type, brick_size, color = self.classifier.parse_classification(top_prediction)
                            
                            if brick_type:
                                print(f"Parsed: Type={brick_type}, Size={brick_size}, Color={color}")
                                
                                # QUEUE the brick instead of processing immediately
                                await self.queue_brick(brick_type, brick_size, current_time)
                                
                                # Update tracking variables
                                last_processed_class = top_prediction
                                last_processing_time = current_time
                                
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
        finally:
            # Cancel conveyor management task
            if self.conveyor_task:
                self.conveyor_task.cancel()
                try:
                    await self.conveyor_task
                except asyncio.CancelledError:
                    pass
            
            self.dashboard.stop()
        
        # Emergency stop if conveyor is still running
        if self.hardware.conveyor_running:
            print("Emergency stop - stopping conveyor...")
            await self.hardware.send_command('9')
        
        print("Automatic sorting stopped.")

    async def queue_brick(self, brick_type, brick_size, detection_time):
        """Queue a brick for processing with immediate timer start"""
        brick_info = {
            'type': brick_type,
            'size': brick_size,
            'detection_time': detection_time,
            'processed': False
        }
        
        self.brick_queue.append(brick_info)
        print(f"Brick queued: {brick_type} {brick_size or 'unknown'} (queue length: {len(self.brick_queue)})")
        
        # Calculate when this brick's processing will complete
        delays = self.strategy.get_pusher_delays(brick_size)
        if brick_type == "damaged":
            processing_end_time = detection_time + max(delays['pusher1'], delays['pusher2']) + 0.5
        else:  # undamaged
            processing_end_time = detection_time + 3.0
        
        # Update conveyor stop time to accommodate this brick
        if processing_end_time > self.conveyor_stop_time:
            self.conveyor_stop_time = processing_end_time
            print(f"Conveyor stop time updated to: {self.conveyor_stop_time:.1f}")

    async def manage_conveyor_and_pushers(self):
        """Manage conveyor operations and pusher timing for all queued bricks"""
        conveyor_started = False
        
        while self.running:
            current_time = time.time()
            
            # Check if we have queued bricks and need to start conveyor
            if self.brick_queue and not conveyor_started:
                print("Starting conveyor for queued bricks...")
                await self.hardware.send_command('8')
                conveyor_started = True
            
            # Process any bricks whose pusher timing has arrived
            for brick_info in self.brick_queue:
                if not brick_info['processed']:
                    await self.check_and_fire_pushers(brick_info, current_time)
            
            # Check if we should stop the conveyor
            if conveyor_started and current_time >= self.conveyor_stop_time and self.brick_queue:
                print("Stopping conveyor - all queued bricks processed...")
                await self.hardware.send_command('9')
                conveyor_started = False
                
                # Clear processed bricks from queue
                self.brick_queue = [brick for brick in self.brick_queue if not brick['processed']]
                
                # Reset stop time
                self.conveyor_stop_time = 0
            
            await asyncio.sleep(0.1)  # Check frequently for precise timing

    async def check_and_fire_pushers(self, brick_info, current_time):
        """Check if it's time to fire pushers for a specific brick"""
        if brick_info['processed']:
            return
        
        detection_time = brick_info['detection_time']
        brick_type = brick_info['type']
        brick_size = brick_info['size']
        
        if brick_type == "damaged":
            delays = self.strategy.get_pusher_delays(brick_size)
            pusher1_time = detection_time + delays['pusher1']
            pusher2_time = detection_time + delays['pusher2']
            
            # Check pusher 1
            if not brick_info.get('pusher1_fired', False) and current_time >= pusher1_time:
                size_display = brick_size if brick_size else "unknown size"
                print(f"Firing pusher 1 for {brick_type} {size_display} brick")
                await self.hardware.send_command('1')
                brick_info['pusher1_fired'] = True
            
            # Check pusher 2
            if not brick_info.get('pusher2_fired', False) and current_time >= pusher2_time:
                size_display = brick_size if brick_size else "unknown size"
                print(f"Firing pusher 2 for {brick_type} {size_display} brick")
                await self.hardware.send_command('4')
                brick_info['pusher2_fired'] = True
            
            # Mark as processed when both pushers have fired
            if brick_info.get('pusher1_fired', False) and brick_info.get('pusher2_fired', False):
                brick_info['processed'] = True
                print(f"Damaged {size_display} brick processing complete!")
        
        elif brick_type == "undamaged":
            # For undamaged bricks, just mark as processed after 3 seconds
            pass_through_time = detection_time + 3.0
            if current_time >= pass_through_time:
                brick_info['processed'] = True
                print(f"Undamaged brick passed through successfully!")

    async def shutdown_system(self):
        await self.hardware.disconnect()
        self.classifier.cleanup()