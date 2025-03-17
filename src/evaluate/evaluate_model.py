import os
import torch
import yaml
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

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

def load_config(config_path="config.yml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _to_predictions(outputs, classification_mode):
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
    config = load_config(config_path)
    classification_mode = config["training"]["classification_mode"]
    classification_enum = ClassificationMode(classification_mode)

    disease = config["training"]["disease"]
    subfolder = config["training"]["selected_tensor_subfolder"]
    discs_for_training = config["training"].get("discs_for_training", ["L4/L5"])

    # --- Load merged CSV and process ---
    import pandas as pd
    raw_csv = os.path.join(config["data"]["raw_path"], "merged_train_data.csv")
    df_all = pd.read_csv(raw_csv)

    df_all["disc_label"] = df_all["level"].str.replace("/", "")

    # Filter to only rows with discs we want
    df_filtered = df_all[df_all["level"].isin(discs_for_training)].copy()

    # Remove duplicates based on study_id and level
    df_filtered = df_filtered.drop_duplicates(subset=["study_id", "level"]).copy()

    if df_filtered.empty:
        print("No data found for requested discs. Exiting.")
        return

    print(f"After filtering to discs {discs_for_training} and removing duplicates, shape={df_filtered.shape}")

    if classification_enum in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        label_col = f"{disease}_label"
        df_filtered = df_filtered.dropna(subset=[label_col])
        df_filtered[label_col] = df_filtered[label_col].astype(int)
        final_label_key = label_col
    else:
        needed_cols = ["scs_label", "lnfn_label", "rnfn_label"]
        df_filtered = df_filtered.dropna(subset=needed_cols)
        for c in needed_cols:
            df_filtered[c] = df_filtered[c].astype(int)
        df_filtered["multi_label"] = (df_filtered["scs_label"].astype(str) + "_" +
                                      df_filtered["lnfn_label"].astype(str) + "_" +
                                      df_filtered["rnfn_label"].astype(str))
        final_label_key = "multi_label"

    train_df, val_df, test_df = create_data_splits(df_filtered, final_label_key,
                                                    config["training"]["test_size"],
                                                    config["training"]["validation_split_of_temp"],
                                                    config["project"]["seed"])

    # We use the test set for evaluation
    # Build disc_dirs mapping:
    interim_base_path = config["data"].get("interim_base_path", "./data/interim")
    disc_dirs = {}
    for d in discs_for_training:
        key = d.replace("/", "")
        disc_dirs[key] = os.path.join(interim_base_path, key, subfolder)

    # Build dataset
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

    loader = torch.utils.data.DataLoader(dataset, batch_size=config["training"]["batch_size"],
                                           shuffle=False, collate_fn=custom_collate_filter_none)

    # --- Build and load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(config, in_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

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

    print("[INFO] Overall metrics:")
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)
    print("Balanced Accuracy:", bal_acc)
    print("Accuracy:", acc)
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, zero_division=0, digits=4))
    print("Imbalanced Classification Report:")
    print(classification_report_imbalanced(all_targets, all_preds, zero_division=0, digits=4))
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:\n", cm)

    # Per-disc breakdown
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
        disc_cm = confusion_matrix(disc_targets, disc_preds)
        print("Confusion Matrix:\n", disc_cm)

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "best_model.pth"
    evaluate_model(model_path)
