import os
import csv
import logging
from datetime import datetime

class TrainingLogger:
    """
    Logs epoch, train/val loss, train/val accuracy, etc., 
    into a CSV file. Also creates a dedicated folder for 
    saved models and logs.
    """
    def __init__(self, model_name_prefix="model_logs"):
        """
        model_name_prefix (str): e.g. "advanced_single_multiclass"
        Creates a subdirectory with a datetime stamp in ./models/.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"./models/{model_name_prefix}_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_path = os.path.join(self.log_dir, "training_log.csv")
        self.fieldnames = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
        
        # Create CSV file with header
        with open(self.csv_path, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
        
        # In-memory storage, optional
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
    
    def log(self, epoch, train_loss, val_loss, train_acc=None, val_acc=None):
        row_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc if train_acc is not None else -1,
            "val_acc": val_acc   if val_acc   is not None else -1
        }
        # Append row to CSV
        with open(self.csv_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(row_data)
        
        # Also store in memory
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)

    def get_log_dir(self):
        """Returns the directory where logs and models are saved."""
        return self.log_dir


def setup_python_logger(logger: logging.Logger, log_file_path: str):
    """
    Attach a file handler to the provided logger, so that 
    all logging messages go to console and to a file at log_file_path.
    """
    # Remove existing handlers to avoid duplicate logs
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file_path, mode='a')
    fh.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
