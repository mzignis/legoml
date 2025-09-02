import multiprocessing as mp
import json
import time
from pathlib import Path
import subprocess
import os

class DashboardConnector:
    def __init__(self, data_file="dashboard_data.json", snapshots_dir="snapshots"):
        self.data_file = Path(data_file)
        self.snapshots_dir = Path(snapshots_dir)
        self.dashboard_process = None
        

    def start_dashboard_process(self):
        print("Starting dashboard process...")
        self.dashboard_process = mp.Process(
            target=self._run_dashboard,
            daemon=True
        )
        self.dashboard_process.start()
        print(f"Dashboard process started with PID: {self.dashboard_process.pid}")
        print("Dashboard should open automatically in your browser!")
        

    def _run_dashboard(self):
        try:
            # Set environment variables to disable email prompt but allow browser
            env = os.environ.copy()
            env.update({
                'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
                'STREAMLIT_SERVER_HEADLESS': 'false',  # Allow browser to open
                'STREAMLIT_SERVER_ENABLE_CORS': 'false',
                'STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION': 'false'
            })
            
            # Run streamlit with configuration flags
            subprocess.run([
                "streamlit", "run", 
                "/home/candfpi4b/fresh_repo/legoml/inference/inference_system_v1/dashboard_v1.py",
                "--server.port=8501",
                "--server.address=localhost",  # Local access only
                "--browser.gatherUsageStats=false",  # Disable usage stats
                #"--global.disableWatchdogWarning=true",  # Disable watchdog warnings
                "--server.runOnSave=true"  # Auto-reload on file changes
            ], env=env, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Dashboard process failed: {e}")
        except FileNotFoundError:
            print("Error: Streamlit not found. Install with: pip install streamlit")
        except Exception as e:
            print(f"Dashboard error: {e}")
    

    def update_data(self, classes, confidences):
        try:
            data = {
                'timestamp': time.time(),
                'classes': classes,
                'confidences': confidences
            }
            
            # Write to shared file (dashboard reads this)
            with open(self.data_file, 'w') as f:
                json.dump(data, f)
                
        except Exception as e:
            print(f"Dashboard update error: {e}")
    
    def stop(self):
        if self.dashboard_process and self.dashboard_process.is_alive():
            print("Stopping dashboard process...")
            self.dashboard_process.terminate()
            self.dashboard_process.join(timeout=5)  # Wait up to 5 seconds
            if self.dashboard_process.is_alive():
                print("Force killing dashboard process...")
                self.dashboard_process.kill()
    
    
    def is_running(self):
        return self.dashboard_process and self.dashboard_process.is_alive()