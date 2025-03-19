import os
import torch
import yaml
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
import pandas as pd

from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report, accuracy_score

from src.model.model_builder import build_model
from src.model.dataset import (
    SingleDiseaseDatasetMultiDisc,
    MultiLabelSpinalDatasetMultiDisc,
    custom_collate_filter_none
)
from src.model.train_model import create_data_splits, parse_depth_from_folder, parse_size_from_folder
from src.utils.enums import ClassificationMode
from src.visualization.visualize import plot_confusion_matrix

def map_label_3class_to_2class(label, mode="normal_negative"):
    """
    Convert a 3-class label to a 2-class label for binary classification.
    
    Args:
        label: int, original label (0, 1, or 2).
        mode: str, mapping mode; "severe_positive" maps 2 to 1 and others to 0,
                   "normal_negative" maps 0 to 0 and others to 1.
    
    Returns:
        int: Binary-mapped label.
    """
    if mode == "severe_positive":
        return 1 if label == 2 else 0
    else:
        return 0 if label == 0 else 1

def load_config(config_path="config.yml") -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: str, path to the configuration file.
    
    Returns:
        dict: Loaded configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _to_predictions(outputs, classification_mode):
    """
    Convert raw model outputs into predicted class labels based on the classification mode.
    
    Args:
        outputs: torch.Tensor, raw outputs from the model.
        classification_mode: str, mode of classification (e.g., "single_multiclass", "single_binary").
    
    Returns:
        torch.Tensor: Predicted class labels.
    """
    if classification_mode == "single_multiclass":
        return torch.argmax(outputs, dim=1)
    elif classification_mode == "single_binary":
        probs = torch.sigmoid(outputs)
        return (probs > 0.5).float().squeeze(1)
    elif classification_mode == "multi_multiclass":
        out_scs, out_lnfn, out_rnfn = outputs
        scs_pred  = torch.argmax(out_scs, dim=1, keepdim=True)
        lnfn_pred = torch.argmax(out_lnfn, dim=1, keepdim=True)
        rnfn_pred = torch.argmax(out_rnfn, dim=1, keepdim=True)
        return torch.cat([scs_pred, lnfn_pred, rnfn_pred], dim=1)
    else:  # multi_binary
        out_scs, out_lnfn, out_rnfn = outputs
        scs_probs  = torch.sigmoid(out_scs)
        lnfn_probs = torch.sigmoid(out_lnfn)
        rnfn_probs = torch.sigmoid(out_rnfn)
        scs_pred  = (scs_probs > 0.5).float().unsqueeze(1)
        lnfn_pred = (lnfn_probs > 0.5).float().unsqueeze(1)
        rnfn_pred = (rnfn_probs > 0.5).float().unsqueeze(1)
        return torch.cat([scs_pred, lnfn_pred, rnfn_pred], dim=1)

def evaluate_model(model_path: str, config_path="config.yml", split="test"):
    """
    Evaluate the trained model on the test set.
    
    This function supports two evaluation modes:
      - When a saved test split CSV exists and/or 'use_saved_test_split' is True,
        only the data from the CSV is used.
      - Otherwise, the full dataset is loaded and filtered by the evaluation discs.
        For each disc, if saved test data exist for that disc, they are used; otherwise,
        the full subset is used.
    
    Args:
        model_path: str, path to the saved model checkpoint.
        config_path: str, path to the configuration YAML file.
        split: str, name of the split being evaluated (default "test").
    
    Returns:
        None. Prints evaluation metrics to stdout.
    """
    config = load_config(config_path)
    classification_mode = config["training"]["classification_mode"]
    classification_enum = ClassificationMode(classification_mode)
    disease = config["training"]["disease"]
    subfolder = config["training"]["selected_tensor_subfolder"]

    # Determine whether to use a saved test split.
    use_saved_test_split = config["evaluation"].get("use_saved_test_split", True)
    model_dir = os.path.dirname(model_path)
    saved_split_path = os.path.join(model_dir, "test_split.csv")
    
    if use_saved_test_split:
        if not os.path.exists(saved_split_path):
            raise FileNotFoundError(f"Test split file not found at {saved_split_path}")
        test_df = pd.read_csv(saved_split_path)
        print(f"[INFO] Loaded saved test split from: {saved_split_path}")
    else:
        # Optionally load the saved test split if it exists.
        if os.path.exists(saved_split_path):
            saved_df = pd.read_csv(saved_split_path)
            print(f"[INFO] Found saved test split at: {saved_split_path}")
        else:
            saved_df = pd.DataFrame()
        
        # Load full dataset and filter by evaluation discs.
        full_dataset_path = config["evaluation"].get(
            "full_dataset_path",
            os.path.join(config["data"]["raw_path"], "merged_train_data.csv")
        )
        full_df = pd.read_csv(full_dataset_path)
        full_df["disc_label"] = full_df["level"].str.replace("/", "")
        
        # Filter full dataset to only include evaluation discs.
        discs_for_evaluation = config["evaluation"].get(
            "discs_for_evaluation",
            config["training"].get("discs_for_training", [])
        )
        if discs_for_evaluation:
            full_df = full_df[full_df["level"].isin(discs_for_evaluation)].copy()
            print(f"[INFO] Filtering full dataset for evaluation discs: {discs_for_evaluation}")
        else:
            print("[INFO] No discs_for_evaluation specified; using the full dataset.")
        
        # Remove duplicates.
        full_df = full_df.drop_duplicates(subset=["study_id", "level"]).copy()
        
        # Prepare labels and apply binary mapping if necessary.
        if classification_enum in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
            label_col = f"{disease}_label"
            full_df = full_df.dropna(subset=[label_col])
            full_df[label_col] = full_df[label_col].astype(int)
            if classification_enum == ClassificationMode.SINGLE_BINARY:
                binary_mapping_mode = config["training"].get("binary_mapping_mode", "normal_negative")
                full_df[label_col] = full_df[label_col].apply(lambda x: map_label_3class_to_2class(x, mode=binary_mapping_mode))
            final_label_key = label_col
        else:
            needed_cols = ["scs_label", "lnfn_label", "rnfn_label"]
            full_df = full_df.dropna(subset=needed_cols)
            for c in needed_cols:
                full_df[c] = full_df[c].astype(int)
            full_df["multi_label"] = (full_df["scs_label"].astype(str) + "_" +
                                      full_df["lnfn_label"].astype(str) + "_" +
                                      full_df["rnfn_label"].astype(str))
            final_label_key = "multi_label"
        
        # For each evaluation disc, prefer using saved data if available.
        combined_list = []
        for disc in discs_for_evaluation:
            disc_saved = saved_df[saved_df["level"] == disc] if not saved_df.empty else pd.DataFrame()
            if not disc_saved.empty:
                print(f"[INFO] Model was trained on data for disc {disc}; using saved test split data for evaluation.")
                combined_list.append(disc_saved)
            else:
                disc_full = full_df[full_df["level"] == disc]
                print(f"[INFO] No saved test split data for disc {disc}; using full dataset subset for evaluation.")
                combined_list.append(disc_full)
        test_df = pd.concat(combined_list, ignore_index=True)
        print(f"[INFO] Combined evaluation test set size: {len(test_df)}")
    
    # --- Build disc_dirs mapping ---
    # Construct a mapping from disc names (without slashes) to their data directories.
    discs_to_use = config["evaluation"].get("discs_for_evaluation",
                      config["training"].get("discs_for_training", ["L4/L5"]))
    interim_base_path = config["data"].get("interim_base_path", "./data/interim")
    disc_dirs = {}
    for d in discs_to_use:
        key = d.replace("/", "")
        disc_dirs[key] = os.path.join(interim_base_path, key, subfolder)
    
    # --- Build dataset based on classification mode ---
    if classification_enum in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        dataset = SingleDiseaseDatasetMultiDisc(
            test_df, disc_dirs, disease,
            parse_depth_from_folder(subfolder),
            parse_size_from_folder(subfolder),
            classification_mode=classification_mode
        )
        in_channels = 1
    else:
        dataset = MultiLabelSpinalDatasetMultiDisc(
            test_df, disc_dirs,
            parse_depth_from_folder(subfolder),
            parse_size_from_folder(subfolder),
            classification_mode=classification_mode
        )
        in_channels = 3

    # Create DataLoader for evaluation.
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=custom_collate_filter_none
    )

    # --- Build and load the trained model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(config, in_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Collect predictions and ground-truth labels.
    all_targets = []
    all_preds = []
    all_disc_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating on {split} set"):
            if batch is None:
                continue
            x, y, disc_labels_batch = batch
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            preds = _to_predictions(outputs, classification_mode)
            all_targets.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_disc_labels.extend(disc_labels_batch)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # --- Compute and Print Evaluation Metrics ---
    print("[INFO] Overall metrics:")
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)
    print("Accuracy:", acc)
    print("Balanced Accuracy:", bal_acc)
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, zero_division=0, digits=4))
    print("Imbalanced Classification Report:")
    print(classification_report_imbalanced(all_targets, all_preds, zero_division=0, digits=4))
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:\n", cm)

    # --- Per-Disc Breakdown ---
    unique_discs = sorted(set(all_disc_labels))
    for disc in unique_discs:
        indices = [i for i, d in enumerate(all_disc_labels) if d == disc]
        disc_targets = all_targets[indices]
        disc_preds = all_preds[indices]
        disc_bal_acc = balanced_accuracy_score(disc_targets, disc_preds)
        disc_acc = accuracy_score(disc_targets, disc_preds)
        print(f"\n[INFO] Metrics for {disc}:")
        print("Accuracy:", disc_acc)
        print("Balanced Accuracy:", disc_bal_acc)
        print("Classification Report:")
        print(classification_report(disc_targets, disc_preds, zero_division=0, digits=4))
        print("Imbalanced Classification Report:")
        print(classification_report_imbalanced(disc_targets, disc_preds, zero_division=0, digits=4))
        disc_cm = confusion_matrix(disc_targets, disc_preds)
        print("Confusion Matrix:\n", disc_cm)

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "best_model.pth"
    evaluate_model(model_path)
