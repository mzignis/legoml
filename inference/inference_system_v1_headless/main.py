import asyncio
from sorting_orchestrator import SortingOrchestrator

def wait_for_user_input():
    input("Press ENTER after the hub light turns CYAN ")


async def main():
    print("AUTOMATIC BRICK SORTING CONTROLLER")
    print("=" * 50)
    
    # You can customize these parameters
    CLASSIFIER_MODEL_PATH = "/home/candfpi4b/lego_pdm/brick_classifier_simple96.pth"
    INTERVAL = 3.0  # How often classifier makes predictions & how often to check for new predictions(seconds)
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
        
        # Start classifier system
        if not await orchestrator.start_classifier_system(INTERVAL):
            return
        
        print("\nSYSTEM READY!")
        print("The system will now automatically detect and sort bricks!")
        print(f"Prediction & Check interval: {INTERVAL}s")
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print()
        
        # Start automatic sorting
        await orchestrator.automatic_brick_sorting_loop(
            confidence_threshold=CONFIDENCE_THRESHOLD,
            check_interval=INTERVAL
        )
        
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    finally:
        await orchestrator.shutdown_system()
        print("Program ended")


if __name__ == "__main__":
    asyncio.run(main())