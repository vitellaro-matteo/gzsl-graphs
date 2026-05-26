"""Experiment logging. Migrated from logs.py."""

import os, json, csv
from datetime import datetime
from typing import Dict


class ExperimentLogger:
    """Logs results to CSV and JSON.

    Usage:
        logger = ExperimentLogger("results/")
        logger.log(model="dgpn", dataset="cora", i_zsl=0.34)
        logger.save()
    """

    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records = []

    def log(self, **kwargs):
        self.records.append({"timestamp": datetime.now().isoformat(), **kwargs})

    def save(self, filename=None):
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{ts}.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2, default=str)
        return path

    def save_csv(self, filename=None):
        if not self.records: return
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{ts}.csv"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.records[0].keys())
            w.writeheader()
            w.writerows(self.records)
        return path
