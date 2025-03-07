import os
import torch
import yaml
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.visualization.visualize import plot_confusion_matrix
from src.model.dataset import SingleDiseaseDataset, MultiLabelSpinalDataset
from src.model.model_builder import build_model
from src.model.train_model import create_data_splits, parse_depth_from_folder, parse_size_from_folder


def load_config(config_path: str = "config.yml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _to_predictions(outputs, classification_mode):
    """
    Convert logits -> discrete predictions.
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


def evaluate_model(model_path: str, config_path: str = "config.yml", split: str = "test"):
    config = load_config(config_path)
    classification_mode = config["training"]["classification_mode"]
    disease = config["training"]["disease"]
    folder  = config["training"]["selected_tensor_folder"]
    model_arch = config["training"]["model_arch"]
    batch_size   = config["training"]["batch_size"]
    dropout_prob = config["training"]["dropout_prob"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data paths and configuration
    import pandas as pd
    from sklearn.model_selection import train_test_split

    raw_csv   = os.path.join(config["data"]["raw_path"], "merged_train_data.csv")
    interim_base = config["data"]["interim_path"]
    selected_path = os.path.join(interim_base, folder)

    # Use the same parsing functions as training
    final_depth = parse_depth_from_folder(folder)
    final_size  = parse_size_from_folder(folder)

    df = pd.read_csv(raw_csv)
    
    # For multi-disease, ensure that missing labels are dropped before deduplication.
    if classification_mode.startswith("single"):
        label_col = f"{disease}_label"
        df = df.dropna(subset=[label_col])
        df_dedup = df.groupby("study_id", as_index=False).first()
        df_dedup[label_col] = df_dedup[label_col].astype(int)
        # Create the split using the shared helper.
        _, _, test_df = create_data_splits(df_dedup, label_col,
                                           test_size=config["training"]["test_size"],
                                           val_split=config["training"]["validation_split_of_temp"],
                                           seed=config["project"]["seed"])
        input_channels = 1
        dataset = SingleDiseaseDataset(test_df, disease,
                                       os.path.join(selected_path, disease),
                                       final_depth, final_size,
                                       classification_mode=classification_mode)
    else:
        needed_cols = ["scs_label", "lnfn_label", "rnfn_label"]
        # Drop studies missing any of the labels first, then deduplicate.
        df = df.dropna(subset=needed_cols)
        df_dedup = df.groupby("study_id", as_index=False).first()
        df_dedup[needed_cols] = df_dedup[needed_cols].astype(int)
        # Create a combined column for stratification.
        df_dedup["multi_label"] = (df_dedup["scs_label"].astype(str) + "_" +
                                   df_dedup["lnfn_label"].astype(str) + "_" +
                                   df_dedup["rnfn_label"].astype(str))
        _, _, test_df = create_data_splits(df_dedup, "multi_label",
                                           test_size=config["training"]["test_size"],
                                           val_split=config["training"]["validation_split_of_temp"],
                                           seed=config["project"]["seed"])
        input_channels = 3
        scs_dir  = os.path.join(selected_path, "scs")
        lnfn_dir = os.path.join(selected_path, "lnfn")
        rnfn_dir = os.path.join(selected_path, "rnfn")
        dataset = MultiLabelSpinalDataset(test_df, scs_dir, lnfn_dir, rnfn_dir,
                                           final_depth, final_size,
                                           classification_mode)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Build the model using the centralized model builder.
    model, _ = build_model(config, input_channels)
    print(f"[INFO] Loading model from {model_path} using arch={model_arch}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Evaluating on {split} set"):
            x = x.to(device)
            y = y.to(device)

            if classification_mode.startswith("single"):
                outputs = model(x)
                preds = _to_predictions(outputs, classification_mode)
                all_targets.extend(y.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            else:
                out_scs, out_lnfn, out_rnfn = model(x)
                preds = _to_predictions((out_scs, out_lnfn, out_rnfn), classification_mode)
                all_targets.append(y.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

    if classification_mode.startswith("single"):
        all_targets = np.array(all_targets).flatten()
        all_preds   = np.array(all_preds).flatten()
    else:
        all_targets = np.concatenate(all_targets, axis=0)  # Shape: (num_studies, 3)
        all_preds   = np.concatenate(all_preds, axis=0)

    print("[INFO] Evaluation Completed.\n")

    # Reporting metrics
    if classification_mode in ["single_multiclass", "single_binary"]:
        acc = accuracy_score(all_targets, all_preds)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(all_targets, all_preds, zero_division=0))
        cm = confusion_matrix(all_targets, all_preds)
        print("Confusion Matrix:\n", cm)
        plot_confusion_matrix(cm, class_names=None,
                              title=f"{classification_mode} Confusion Matrix")
    else:
        diseases = ["SCS", "LNfN", "RNfN"]
        for i, dname in enumerate(diseases):
            print(f"=== Metrics for {dname} ===")
            t = all_targets[:, i]
            p = all_preds[:, i]
            acc = accuracy_score(t, p)
            print(f"Accuracy: {acc:.4f}")
            print("Classification Report:")
            print(classification_report(t, p, zero_division=0))
            cm = confusion_matrix(t, p)
            print("Confusion Matrix:\n", cm, "\n")
            # Adjust class names as needed:
            class_names = ["0", "1", "2"] if classification_mode == "multi_multiclass" else ["0", "1"]
            plot_confusion_matrix(cm, class_names=class_names,
                                  title=f"{dname} Confusion Matrix")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "best_model.pth"
    evaluate_model(model_path)
