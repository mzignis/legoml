import threading
import time
import os
from pathlib import Path
from typing import Callable, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalListener:
    """Simple threaded signal listener for dashboard updates."""
    
    def __init__(self, signal_file_path: str, check_interval: float = 0.5):
        """
        Initialize the signal listener.
        
        Args:
            signal_file_path: Path to directory containing signal file
            check_interval: How often to check for signal file (seconds)
        """
        self.signal_file = Path(signal_file_path) / ".dashboard_update_signal"
        self.check_interval = check_interval
        self.running = False
        self.callback = None
        self.thread = None
    
    def start(self, callback: Callable[[bool], None]) -> threading.Thread:
        """
        Start continuous listening in background thread.
        
        Args:
            callback: Function to call when signal is detected
            
        Returns:
            Thread object (already started)
        """
        self.callback = callback
        self.running = True
        
        def _listen_loop():
            logger.info(f"Signal listener started: {self.signal_file}")
            
            while self.running:
                try:
                    if self.signal_file.exists():
                        logger.info("Signal detected - processing...")
                        
                        # Delete signal file
                        self.signal_file.unlink()
                        logger.info("Signal file deleted")
                        
                        # Trigger callback
                        if self.callback:
                            self.callback(True)
                    
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in listener: {e}")
                    time.sleep(self.check_interval)
        
        self.thread = threading.Thread(target=_listen_loop, daemon=True)
        self.thread.start()
        return self.thread
    
    def stop(self):
        """Stop the listener."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("Signal listener stopped")


# Simple function for quick setup
def start_signal_listener(signal_path: str, callback: Callable[[bool], None], 
                         check_interval: float = 0.5) -> SignalListener:
    """
    Quick setup function - starts listener and returns the object.
    
    Args:
        signal_path: Path to signal file directory
        callback: Function to call when signal received
        check_interval: Check frequency in seconds
        
    Returns:
        SignalListener object (already running)
    """
    listener = SignalListener(signal_path, check_interval)
    listener.start(callback)
    return listener


# Usage example
if __name__ == "__main__":
    
    def on_dashboard_update(signal: bool):
        """Called when update signal is received."""
        if signal:
            print("🔄 Dashboard update signal received!")
            # Add your dashboard refresh logic here
    
    # Start the listener
    listener = start_signal_listener(
        signal_path="/home/candfpi4b/fresh_repo/legoml/inference/inference_system_v1",
        callback=on_dashboard_update,
        check_interval=0.5
    )
    
    # Your dashboard code continues here
    try:
        print("📊 Dashboard running...")
        while True:
            print("Dashboard working...")
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        listener.stop()