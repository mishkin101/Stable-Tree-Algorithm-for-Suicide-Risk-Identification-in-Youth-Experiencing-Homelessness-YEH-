# src/StableTree/logging_utils.py
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    def __init__(self, experiment_name: str = None, group_name: str = None):
        """Initialize a logger for either a single experiment or an entire group."""
        # Choose base directory: either per-group under experiments/, or per-experiment under logs/
        if group_name:
            # Logging for a group: write directly under experiments/<group_name>
            self.base_dir = Path("experiments") / group_name
        else:
            # Logging for individual experiment: write under logs/
            self.base_dir = Path("logs").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Determine the experiment directory
        if experiment_name:
            # Individual experiment or subgroup logging
            self.experiment_dir = self.base_dir / experiment_name
        else:
            # Group-level logger writes directly into base_dir
            self.experiment_dir = self.base_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create (or reuse) a plots subdirectory
        self.plots_dir = self.experiment_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Initialize metrics and config
        self.metrics = {}
        self.config = {}

        # Create or clear a log file
        self.log_file = self.experiment_dir / "log.txt"
        with open(self.log_file, "w") as f:
            f.write(
                f"Logger started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

    def log_config(self, config_dict: dict):
        """Log configuration parameters."""
        self.config.update(config_dict)
        config_file = self.experiment_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=4)
        with open(self.log_file, "a") as f:
            f.write("=== Configuration ===\n")
            for k, v in config_dict.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    def log_metric(self, name: str, value):
        """Log a single metric."""
        self.metrics[name] = value
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def log_metrics(self, metrics_dict: dict):
        """Log multiple metrics at once."""
        self.metrics.update(metrics_dict)
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)
        with open(self.log_file, "a") as f:
            f.write("=== Metrics ===\n")
            for k, v in metrics_dict.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    def save_figure(self, name: str):
        """Save the current matplotlib figure to the plots directory."""
        fig_path = self.plots_dir / f"{name}.png"
        plt.savefig(fig_path)
        with open(self.log_file, "a") as f:
            f.write(f"Saved figure: {name}.png\n")
        return fig_path

