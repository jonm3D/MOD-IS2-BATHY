"""
Progress monitoring utility for batch processing.
"""

import os
import json
import time
from datetime import datetime, timedelta

class ProgressMonitor:
    """Tracks and reports progress of batch jobs"""
    
    def __init__(self, total_items, output_dir, update_interval=5):
        """
        Initialize a progress monitor.
        
        Parameters:
        -----------
        total_items : int
            Total number of items to process
        output_dir : str
            Directory to save progress reports
        update_interval : int, optional
            Seconds between progress file updates (default: 5)
        """
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.in_progress_items = 0
        self.start_time = datetime.now()
        self.update_interval = update_interval
        self.last_update_time = 0
        
        # Create progress file path
        self.progress_file = os.path.join(output_dir, "progress.json")
        
        # Initialize progress file
        self._update_progress_file()
    
    def item_started(self):
        """Mark an item as started"""
        self.in_progress_items += 1
        self._conditionally_update()
    
    def item_completed(self, success=True):
        """
        Mark an item as completed.
        
        Parameters:
        -----------
        success : bool, optional
            Whether the item completed successfully (default: True)
        """
        self.in_progress_items -= 1
        if success:
            self.completed_items += 1
        else:
            self.failed_items += 1
        self._conditionally_update()
    
    def _conditionally_update(self):
        """Update progress file if enough time has passed"""
        current_time = time.time()
        if (current_time - self.last_update_time) >= self.update_interval:
            self._update_progress_file()
            self.last_update_time = current_time
    
    def _update_progress_file(self):
        """Update the progress file with current stats"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        completed = self.completed_items + self.failed_items
        remaining = self.total_items - completed
        
        # Calculate estimated time remaining
        if completed > 0:
            avg_time_per_item = elapsed_time / completed
            est_time_remaining = avg_time_per_item * remaining
            est_completion_time = datetime.now() + timedelta(seconds=est_time_remaining)
            
            # Format time remaining as HH:MM:SS
            hours, remainder = divmod(est_time_remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_remaining_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            completion_time_str = est_completion_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_remaining_str = "calculating..."
            completion_time_str = "calculating..."
        
        # Calculate percentage
        percent_complete = (completed / self.total_items) * 100 if self.total_items > 0 else 0
        
        progress_data = {
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "in_progress_items": self.in_progress_items,
            "percent_complete": round(percent_complete, 2),
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time": str(timedelta(seconds=int(elapsed_time))),
            "est_time_remaining": time_remaining_str,
            "est_completion_time": completion_time_str,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Write to file
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def finalize(self):
        """Finalize the progress tracking"""
        self._update_progress_file()