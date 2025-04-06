import os
import sys
import yaml
import ast
from src.evaluate.evaluate_model import evaluate_model

def read_base_config(config_path="config.yml"):
    """Read the base configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_training_log(log_path):
    """
    Parse the training.log file to extract key–value parameters.
    Expects lines in the log in the format:
      [timestamp INFO] __main__: key: value
    Returns a dictionary with these key–value pairs.
    """
    params = {}
    with open(log_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            try:
                # Remove the timestamp and bracketed prefix.
                parts = line.split("]")
                if len(parts) < 2:
                    continue
                message = parts[1].strip()
                if message.startswith("__main__:"):
                    message = message[len("__main__:"):].strip()
                if ":" in message:
                    key, value = message.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    # Safely evaluate value (to get numbers, lists, etc.)
                    try:
                        value = ast.literal_eval(value)
                    except Exception:
                        pass
                    params[key] = value
            except Exception:
                continue
    return params

def update_config_with_log(base_config, log_params):
    """
    Update the base configuration dictionary with parameters from the training log.
    Only keys that already exist in base_config["training"] are updated.
    """
    if "training" in base_config:
        for key, value in log_params.items():
            if key in base_config["training"]:
                base_config["training"][key] = value
    return base_config

def save_config(config, path):
    """Save a configuration dictionary to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_from_logs.py <model_checkpoint_path>")
        sys.exit(1)
    # The only required argument is the full path to the model checkpoint.
    model_checkpoint_path = sys.argv[1]
    if not os.path.exists(model_checkpoint_path):
        print(f"Checkpoint not found: {model_checkpoint_path}")
        sys.exit(1)
    
    # Derive the model directory from the checkpoint path.
    model_dir = os.path.dirname(model_checkpoint_path)
    
    # Assume the training log is in the model directory.
    log_path = os.path.join(model_dir, "training.log")
    if not os.path.exists(log_path):
        print(f"Training log not found at {log_path}")
        sys.exit(1)
    
    # Read the base configuration from the default "config.yml".
    base_config = read_base_config("config.yml")
    
    # Parse the training.log for key parameters.
    log_params = parse_training_log(log_path)
    
    # Update the base configuration's "training" section with parameters from the log.
    updated_config = update_config_with_log(base_config, log_params)
    
    # Ensure the evaluation section is set up.
    if "evaluation" not in updated_config:
        updated_config["evaluation"] = {}
    if "discs_for_evaluation" not in updated_config["evaluation"]:
        # Use training discs if evaluation discs are not explicitly provided.
        updated_config["evaluation"]["discs_for_evaluation"] = updated_config["training"].get("discs_for_training", [])
    
    # Save the updated configuration to a temporary file in the model directory.
    temp_config_path = os.path.join(model_dir, "temp_eval_config.yml")
    save_config(updated_config, temp_config_path)
    
    print(f"Evaluating model from: {model_checkpoint_path}")
    print(f"Using configuration from: {temp_config_path}")
    
    # Run the evaluation using the updated configuration.
    evaluate_model(model_checkpoint_path, config_path=temp_config_path, split="test")

if __name__ == "__main__":
    main()
