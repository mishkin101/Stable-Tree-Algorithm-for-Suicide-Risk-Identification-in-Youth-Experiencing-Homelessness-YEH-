# src/StableTree/logging_utils.py
import os
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    def __init__(self, experiment_name=None):
        """Initialize a logger for the experiment."""
        # Create logs directory if it doesn't exist
        self.logs_dir = Path("logs").resolve()
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create a unique experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        # Create a directory for this experiment
        self.experiment_dir = self.logs_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.experiment_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize metrics dictionary
        self.metrics = {}
        
        # Initialize config dictionary
        self.config = {}
        
        # Create a log file
        self.log_file = self.experiment_dir / "log.txt"
        with open(self.log_file, "w") as f:
            f.write(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log_config(self, config_dict):
        """Log configuration parameters."""
        self.config.update(config_dict)
        
        # Save config to a JSON file
        config_file = self.experiment_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=4)
        
        # Also append to the log file
        with open(self.log_file, "a") as f:
            f.write("=== Configuration ===\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    def log_metric(self, name, value):
        """Log a metric value."""
        self.metrics[name] = value
        
        # Update metrics file
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)
    
    def log_metrics(self, metrics_dict):
        """Log multiple metrics at once."""
        self.metrics.update(metrics_dict)
        
        # Update metrics file
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)
        
        # Also append to the log file
        with open(self.log_file, "a") as f:
            f.write("=== Metrics ===\n")
            for key, value in metrics_dict.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    def save_figure(self, name):
        """Save the current matplotlib figure to the plots directory."""
        figure_path = self.plots_dir / f"{name}.png"
        plt.savefig(figure_path)
        
        # Log that a figure was saved
        with open(self.log_file, "a") as f:
            f.write(f"Saved figure: {name}.png\n")
        
        return figure_path