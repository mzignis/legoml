import asyncio
from sorting_orchestrator import SortingOrchestrator

def wait_for_user_input():
    input("Press ENTER after the hub light turns CYAN ")

async def main():
    print("AUTOMATIC BRICK SORTING CONTROLLER")
    print("=" * 50)

    # You can customize these parameters
    CLASSIFIER_MODEL_PATH = "/home/candfpi4b/fresh_repo/legoml/inference/inference_system_v1/brick_classifier_simple_torchscript.pt"
    INTERVAL = 2.5  # Minimum time between processing same class (used for both classifier and orchestrator)
    CHECK_INTERVAL = 1.0  # How often to check for new predictions (can be faster than INTERVAL)
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to process a brick

    orchestrator = SortingOrchestrator(CLASSIFIER_MODEL_PATH)

    try:
        # Initialize all system components
        if not await orchestrator.initialize_system():
            return

        print("\nPress the CENTER BUTTON on your Pybricks hub to start the program")
        print("Press ENTER here when the hub light is cyan")
        print()
        wait_for_user_input()

        # Start classifier system with the same interval used for sorting
        if not await orchestrator.start_classifier_system(INTERVAL, CHECK_INTERVAL):
            return

        print("\nSYSTEM READY!")
        print("The system will now automatically detect and sort bricks!")
        print(f"Same class processing interval: {INTERVAL}s")
        print(f"Prediction check interval: {CHECK_INTERVAL}s") 
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print()

        # Start automatic sorting - both intervals use INTERVAL for consistency
        await orchestrator.automatic_brick_sorting_loop(
            confidence_threshold=CONFIDENCE_THRESHOLD,
            check_interval=CHECK_INTERVAL,
            min_processing_interval=INTERVAL  # Same as classifier interval
        )

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        await orchestrator.shutdown_system()
        print("Program ended")

if __name__ == "__main__":  # Fixed the typo here
    asyncio.run(main())