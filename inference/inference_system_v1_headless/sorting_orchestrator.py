import asyncio
from hardware_controller import HardwareController
from classifier_manager import ClassifierManager
from sorting_strategy import SortingStrategy


class SortingOrchestrator:
    def __init__(self, classifier_model_path):
        self.hardware = HardwareController()
        self.classifier = ClassifierManager(classifier_model_path)
        self.strategy = SortingStrategy(self.hardware)
        self.running = False


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
        return True


    async def start_classifier_system(self, prediction_interval):
        print("\nStep 3: Starting brick classifier...")
        if not self.classifier.start_classifier(prediction_interval):
            print("Failed to start classifier. Exiting.")
            return False
        return True

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
                if self.hardware.conveyor_running:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get latest predictions from classifier
                try:
                    classes, confidences = self.classifier.get_latest_predictions()
                    
                    if classes and confidences:
                        top_prediction = classes[0]
                        top_confidence = confidences[0]
                        
                        # Only process if confidence is above threshold and it's a new prediction
                        if top_confidence >= confidence_threshold and top_prediction != last_processed_prediction:
                            print(f"New detection: {top_prediction} (confidence: {top_confidence:.3f})")
                            
                            # Parse the classification
                            brick_type, brick_size, color = self.classifier.parse_classification(top_prediction)
                            
                            if brick_type:
                                print(f"Parsed: Type={brick_type}, Size={brick_size}, Color={color}")
                                
                                # Process the brick using strategy
                                await self.strategy.process_brick(brick_type, brick_size)
                                
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
        if self.hardware.conveyor_running:
            print("Emergency stop - stopping conveyor...")
            await self.hardware.send_command('9')
        
        print("Automatic sorting stopped.")


    async def shutdown_system(self):
        await self.hardware.disconnect()
        self.classifier.cleanup()