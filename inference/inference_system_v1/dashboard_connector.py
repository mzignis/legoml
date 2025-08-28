import multiprocessing as mp
import json
import time
from pathlib import Path
import subprocess

class DashboardConnector:
    def __init__(self, data_file="dashboard_data.json", snapshots_dir="snapshots"):
        self.data_file = Path(data_file)
        self.snapshots_dir = Path(snapshots_dir)
        self.dashboard_process = None
        

    def start_dashboard_process(self):
        self.dashboard_process = mp.Process(
            target=self._run_dashboard,
            daemon=True
        )
        self.dashboard_process.start()
        

    def _run_dashboard(self):
        subprocess.run([
            "streamlit", "run", 
            "/home/candfpi4b/lego_pdm/legoml/inference/inference_system_v1/dashboard_v1.py",
            "--server.port=8501"
        ])
        
    
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
            self.dashboard_process.terminate()