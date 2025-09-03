import json
import time
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib
import multiprocessing as mp


class UpdateDetector:
    """
    Simple update detector that monitors a JSON file and JPG folder.
    Sends a file-based signal when any changes are detected.
    """
    
    def __init__(
        self,
        json_path: str,
        snapshots_folder: str,
        polling_interval: float = 0.5,
        use_watchdog: bool = True
    ):
        """
        Initialize the UpdateDetector.
        
        Args:
            json_path: Path to the JSON file to monitor
            snapshots_folder: Path to the folder containing JPG files
            polling_interval: How often to check for changes (if not using watchdog)
            use_watchdog: Whether to use watchdog for efficient file monitoring
        """
        self.json_path = Path(json_path)
        self.snapshots_folder = Path(snapshots_folder)
        self.polling_interval = polling_interval
        self.use_watchdog = use_watchdog
        
        # State tracking
        self.json_content_hash = ""
        self.jpg_files_state = {}  # filename -> (size, mtime)
        self.is_running = False
        self._stop_event = threading.Event()
        
        # Watchdog components
        self.observer = None
        self.file_handler = None
        
        # Threading
        self.monitor_thread = None
        
        # Initialize state
        self._update_json_state()
        self._update_jpg_state()
    

    def _update_json_state(self):
        """Update the tracked state of the JSON file."""
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    content = f.read()
                    self.json_content_hash = hashlib.md5(content.encode()).hexdigest()
            except (OSError, json.JSONDecodeError):
                pass
    

    def _update_jpg_state(self):
        """Update the tracked state of JPG files."""
        if self.snapshots_folder.exists():
            self.jpg_files_state = {}
            for jpg_file in self.snapshots_folder.glob("*.jpg"):
                try:
                    stat = jpg_file.stat()
                    self.jpg_files_state[jpg_file.name] = (stat.st_size, stat.st_mtime)
                except OSError:
                    pass
    

    def _check_for_changes(self) -> bool:
        """Check if any changes occurred. Returns True if changes detected."""
        changes_detected = False
        
        # Check JSON file
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    content = f.read()
                    current_hash = hashlib.md5(content.encode()).hexdigest()
                
                if current_hash != self.json_content_hash:
                    print(f"🔄 JSON CHANGE DETECTED!")
                    print(f"   Old hash: {self.json_content_hash[:8]}...")
                    print(f"   New hash: {current_hash[:8]}...")
                    print(f"   Content preview: {content[:100]}...")
                    self.json_content_hash = current_hash
                    changes_detected = True
                else:
                    print(f"📄 JSON checked - no changes (hash: {current_hash[:8]}...)")
            except (OSError, json.JSONDecodeError) as e:
                print(f"❌ JSON read error: {e}")
        else:
            print(f"❌ JSON file doesn't exist: {self.json_path}")
        
        # Check JPG files
        if self.snapshots_folder.exists():
            current_files = {}
            
            for jpg_file in self.snapshots_folder.glob("*.jpg"):
                try:
                    stat = jpg_file.stat()
                    current_files[jpg_file.name] = (stat.st_size, stat.st_mtime)
                    
                    # Check if this is a new file or if it has changed
                    if (jpg_file.name not in self.jpg_files_state or 
                        self.jpg_files_state[jpg_file.name] != (stat.st_size, stat.st_mtime)):
                        changes_detected = True
                except OSError:
                    pass
            
            self.jpg_files_state = current_files
        
        return changes_detected
    

    def _send_update_signal(self):
        """Send update signal via file for Streamlit integration."""
        try:
            # Create a signal file that the dashboard can detect
            signal_file = Path("/home/candfpi4b/fresh_repo/legoml/inference/inference_system_v1/.dashboard_update_signal")
            signal_file.touch()
            print(f"📡 Update signal sent at {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Error sending update signal: {e}")
    

    def _polling_monitor(self):
        """Polling-based monitoring loop."""
        while not self._stop_event.is_set():
            if self._check_for_changes():
                self._send_update_signal()
            
            self._stop_event.wait(self.polling_interval)
    

    def start(self):
        """Start monitoring for changes."""
        if self.is_running:
            print("UpdateDetector is already running")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        if self.use_watchdog:
            self._start_watchdog_monitor()
        else:
            self._start_polling_monitor()
    

    def _start_polling_monitor(self):
        """Start polling-based monitoring."""
        self.monitor_thread = threading.Thread(target=self._polling_monitor, daemon=True)
        self.monitor_thread.start()
        print(f"Started polling monitor (interval: {self.polling_interval}s)")
    

    def _start_watchdog_monitor(self):
        """Start watchdog-based monitoring."""
        try:
            self.file_handler = UpdateFileHandler(self)
            self.observer = Observer()
            
            # Watch the JSON file's directory
            if self.json_path.parent.exists():
                self.observer.schedule(self.file_handler, str(self.json_path.parent), recursive=False)
            
            # Watch the snapshots folder
            if self.snapshots_folder.exists():
                self.observer.schedule(self.file_handler, str(self.snapshots_folder), recursive=False)
            
            self.observer.start()
            print("Started watchdog monitor")
            
        except ImportError:
            print("Watchdog not available, falling back to polling")
            self.use_watchdog = False
            self._start_polling_monitor()
    
    
    def stop(self):
        """Stop monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        if self.use_watchdog and self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        print("UpdateDetector stopped")


class UpdateFileHandler(FileSystemEventHandler):
    """File system event handler for watchdog."""
    
    def __init__(self, detector: UpdateDetector):
        self.detector = detector
        super().__init__()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's our JSON file or a JPG file
        if file_path == self.detector.json_path or file_path.suffix.lower() == '.jpg':
            # Add a small delay to handle rapid successive changes
            time.sleep(0.1)
            
            if self.detector._check_for_changes():
                self.detector._send_update_signal()
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Handle new JPG files
        if file_path.suffix.lower() == '.jpg' and file_path.parent == self.detector.snapshots_folder:
            time.sleep(0.1)  # Wait for file to be fully written
            if self.detector._check_for_changes():
                self.detector._send_update_signal()


def monitor_process(json_path, snapshots_folder):
    """
    Monitor process function - run this as your third process.
    
    Args:
        json_path: Path to JSON file to monitor
        snapshots_folder: Path to snapshots folder to monitor
    """
    print("Monitor process started")
    
    detector = UpdateDetector(
        json_path=json_path,
        snapshots_folder=snapshots_folder,
        use_watchdog=True,
        polling_interval=0.5
    )
    
    try:
        detector.start()
        print(f"Monitoring: {json_path} and {snapshots_folder}")
        
        # Keep the monitor process alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        detector.stop()
        print("Monitor process stopped")


if __name__ == "__main__":
    def test_dashboard():
        """Test dashboard that checks for update signals."""
        print("Test dashboard started, monitoring for update signals...")
        
        signal_file = Path(".dashboard_update_signal")
        
        while True:
            try:
                if signal_file.exists():
                    # Remove the signal file and process update
                    signal_file.unlink()
                    print("DASHBOARD: Update detected! Refreshing...")
                    # Your dashboard update logic here
                
                time.sleep(1)  # Check every second
                
            except KeyboardInterrupt:
                break
    
    # Test the system
    JSON_PATH = "/home/candfpi4b/fresh_repo/legoml/dashboard_data.json"
    SNAPSHOTS_FOLDER = "/home/candfpi4b/fresh_repo/snapshots"
    
    # Start monitor
    monitor_proc = mp.Process(target=monitor_process, args=(JSON_PATH, SNAPSHOTS_FOLDER))
    monitor_proc.start()
    
    try:
        print("System running. Modify files to test. Press Ctrl+C to stop.")
        test_dashboard()
    except KeyboardInterrupt:
        print("Shutting down...")
        monitor_proc.terminate()
        monitor_proc.join()