import os
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from sklearn.model_selection import train_test_split

# Local imports
from src.model.dataset import SingleDiseaseDataset, MultiLabelSpinalDataset
from src.utils.logger import TrainingLogger
from src.visualization.visualize import plot_training_curves
from src.model.model_builder import build_model


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_depth_from_folder(folder_name):
    """
    Attempt to parse the depth from a folder name that has e.g. '5D' in the 4th element.
    Example: 'target_window_128x128_5D_B2A2' => parse '5D' => 5
    """
    parts = folder_name.split("_")
    if len(parts) >= 4 and parts[3].endswith("D"):
        depth_str = parts[3]  # e.g. '5D'
        return int(depth_str.replace("D", ""))
    return None


def parse_size_from_folder(folder_name):
    """
    Attempt to parse the resolution (width and height) from a folder name that has e.g. '128x128' 
    in the 3rd element.
    
    Example:
        'target_window_128x128_5D_B2A2' => parse '128x128' => (128, 128)
    
    Returns:
        tuple: (width, height) if found, otherwise None.
    """
    parts = folder_name.split("_")
    if len(parts) >= 4:
        resolution_str = parts[2]  # e.g. '128x128'
        return tuple(map(int, resolution_str.split("x")))
    return None


def create_data_splits(df, label_key, test_size, val_split, seed):
    # Check distribution and perform stratified split if possible
    min_count = df[label_key].value_counts().min()
    if min_count < 2:
        train_df, temp_df = train_test_split(
            df, test_size=test_size, shuffle=True, random_state=seed
        )
    else:
        train_df, temp_df = train_test_split(
            df, test_size=test_size, stratify=df[label_key], shuffle=True, random_state=seed
        )
    min_count_temp = temp_df[label_key].value_counts().min()
    if min_count_temp < 2:
        val_df, test_df = train_test_split(
            temp_df, test_size=val_split, shuffle=True, random_state=seed
        )
    else:
        val_df, test_df = train_test_split(
            temp_df, test_size=val_split, stratify=temp_df[label_key], shuffle=True, random_state=seed
        )
    return train_df, val_df, test_df


def compute_accuracy(outputs, targets, classification_mode):
    """
    Compute accuracy for different classification modes.
    """
    if classification_mode == "single_multiclass":
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        return correct / len(targets)

    elif classification_mode == "single_binary":
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float().squeeze(1)
        correct = (preds == targets).sum().item()
        return correct / len(targets)

    elif classification_mode == "multi_multiclass":
        out_scs, out_lnfn, out_rnfn = outputs
        bsz = targets.shape[0]
        scs_preds = torch.argmax(out_scs, dim=1)
        lnfn_preds = torch.argmax(out_lnfn, dim=1)
        rnfn_preds = torch.argmax(out_rnfn, dim=1)
        scs_true = targets[:, 0]
        lnfn_true = targets[:, 1]
        rnfn_true = targets[:, 2]
        scs_corr = (scs_preds == scs_true).sum().item()
        lnfn_corr = (lnfn_preds == lnfn_true).sum().item()
        rnfn_corr = (rnfn_preds == rnfn_true).sum().item()
        return (scs_corr + lnfn_corr + rnfn_corr) / (3 * bsz)

    else:  # multi_binary
        out_scs, out_lnfn, out_rnfn = outputs
        bsz = targets.shape[0]
        scs_preds = (torch.sigmoid(out_scs) > 0.5).float().squeeze(1)
        lnfn_preds = (torch.sigmoid(out_lnfn) > 0.5).float().squeeze(1)
        rnfn_preds = (torch.sigmoid(out_rnfn) > 0.5).float().squeeze(1)
        scs_true = targets[:, 0]
        lnfn_true = targets[:, 1]
        rnfn_true = targets[:, 2]
        scs_corr = (scs_preds == scs_true).sum().item()
        lnfn_corr = (lnfn_preds == lnfn_true).sum().item()
        rnfn_corr = (rnfn_preds == rnfn_true).sum().item()
        return (scs_corr + lnfn_corr + rnfn_corr) / (3 * bsz)


def early_stopping_check(val_loss, best_val_loss, patience_counter, patience_limit):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    return best_val_loss, patience_counter, (patience_counter >= patience_limit)


def train_model(config_path="config.yml"):
    config = load_config(config_path)
    seed = config["project"]["seed"]
    classification_mode = config["training"]["classification_mode"]
    disease = config["training"]["disease"]
    folder = config["training"]["selected_tensor_folder"]
    model_arch = config["training"]["model_arch"]
    
    test_size = config["training"]["test_size"]
    val_split = config["training"]["validation_split_of_temp"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    lr = config["training"]["learning_rate"]
    dropout_prob = config["training"]["dropout_prob"]
    early_stopping_patience = config["training"]["early_stopping_patience"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data paths and parsing
    raw_csv = os.path.join(config["data"]["raw_path"], "merged_train_data.csv")
    interim_base = config["data"]["interim_path"]
    selected_path = os.path.join(interim_base, folder)

    final_depth = parse_depth_from_folder(folder)
    final_size = parse_size_from_folder(folder)

    # Read and deduplicate
    df = pd.read_csv(raw_csv)
    df_dedup = df.groupby("study_id", as_index=False).first()

    excluded_ids = config["training"].get("excluded_studies", [])
    if excluded_ids:
        initial_len = len(df_dedup)
        df_dedup = df_dedup[~df_dedup["study_id"].isin(excluded_ids)]
        print(f"[INFO] Excluding {len(excluded_ids)} problematic study_ids: {excluded_ids}")
        print(f"[INFO] Filtered df from {initial_len} -> {len(df_dedup)} rows")

    # Build dataset based on classification mode
    if classification_mode.startswith("single"):
        label_col = f"{disease}_label"
        df_dedup = df_dedup.dropna(subset=[label_col])
        df_dedup[label_col] = df_dedup[label_col].astype(int)
        train_df, val_df, test_df = create_data_splits(df_dedup, label_col, test_size, val_split, seed)
        tensor_dir = os.path.join(selected_path, disease)
        train_ds = SingleDiseaseDataset(train_df, disease, tensor_dir, final_depth, final_size, classification_mode)
        val_ds   = SingleDiseaseDataset(val_df, disease, tensor_dir, final_depth, final_size, classification_mode)
        test_ds  = SingleDiseaseDataset(test_df, disease, tensor_dir, final_depth, final_size, classification_mode)
        in_channels = 1
    else:
        needed_cols = ["scs_label", "lnfn_label", "rnfn_label"]
        df_dedup = df_dedup.dropna(subset=needed_cols)
        df_dedup[needed_cols] = df_dedup[needed_cols].astype(int)
        df_dedup["multi_label"] = (
            df_dedup["scs_label"].astype(str) + "_" +
            df_dedup["lnfn_label"].astype(str) + "_" +
            df_dedup["rnfn_label"].astype(str)
        )
        train_df, val_df, test_df = create_data_splits(df_dedup, "multi_label", test_size, val_split, seed)
        scs_dir  = os.path.join(selected_path, "scs")
        lnfn_dir = os.path.join(selected_path, "lnfn")
        rnfn_dir = os.path.join(selected_path, "rnfn")
        train_ds = MultiLabelSpinalDataset(train_df, scs_dir, lnfn_dir, rnfn_dir,
                                           final_depth, final_size, classification_mode)
        val_ds   = MultiLabelSpinalDataset(val_df, scs_dir, lnfn_dir, rnfn_dir,
                                           final_depth, final_size, classification_mode)
        test_ds  = MultiLabelSpinalDataset(test_df, scs_dir, lnfn_dir, rnfn_dir,
                                           final_depth, final_size, classification_mode)
        in_channels = 3

    # Quick debug
    debug_vol, debug_lbl = train_ds[0]
    print("[DEBUG] First sample volume shape:", debug_vol.shape)
    print("[DEBUG] First sample label:", debug_lbl)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Build model using the model_builder helper
    model, criterion = build_model(config, in_channels)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n[INFO] classification_mode={classification_mode}")
    if classification_mode.startswith("single"):
        print(f"[INFO] single disease = {disease}")
    print(f"[INFO] model_arch     = {model_arch}")
    print(f"[INFO] final_depth    = {final_depth}, final_size={final_size}")
    print(f"[INFO] Using device   = {device}")
    print(f"[INFO] Train size = {len(train_ds)}, Val size = {len(val_ds)}, Test size = {len(test_ds)}")

    logger = TrainingLogger(f"{model_arch}_{classification_mode}_{folder}_{batch_size}_{num_epochs}_{lr}_{dropout_prob}")
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses   = []
    train_accs   = []
    val_accs     = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_acc = 0.0
        samples_count = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            if classification_mode.startswith("single"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc_batch = compute_accuracy(outputs, targets, classification_mode)
            else:
                out_scs, out_lnfn, out_rnfn = model(inputs)
                if "multiclass" in classification_mode:
                    ls_scs  = criterion(out_scs, targets[:, 0])
                    ls_lnfn = criterion(out_lnfn, targets[:, 1])
                    ls_rnfn = criterion(out_rnfn, targets[:, 2])
                    loss = ls_scs + ls_lnfn + ls_rnfn
                else:
                    ls_scs  = criterion(out_scs, targets[:, 0].unsqueeze(1))
                    ls_lnfn = criterion(out_lnfn, targets[:, 1].unsqueeze(1))
                    ls_rnfn = criterion(out_rnfn, targets[:, 2].unsqueeze(1))
                    loss = ls_scs + ls_lnfn + ls_rnfn
                acc_batch = compute_accuracy((out_scs, out_lnfn, out_rnfn), targets, classification_mode)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_acc += acc_batch * len(inputs)
            samples_count += len(inputs)

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = correct_acc / samples_count

        model.eval()
        val_run_loss = 0.0
        val_corr = 0.0
        val_samples = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs = inputs.to(device)
                targets = targets.to(device)

                if classification_mode.startswith("single"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    acc_batch = compute_accuracy(outputs, targets, classification_mode)
                else:
                    out_scs, out_lnfn, out_rnfn = model(inputs)
                    if "multiclass" in classification_mode:
                        l_scs  = criterion(out_scs, targets[:, 0])
                        l_lnfn = criterion(out_lnfn, targets[:, 1])
                        l_rnfn = criterion(out_rnfn, targets[:, 2])
                        loss = l_scs + l_lnfn + l_rnfn
                    else:
                        l_scs  = criterion(out_scs, targets[:, 0].unsqueeze(1))
                        l_lnfn = criterion(out_lnfn, targets[:, 1].unsqueeze(1))
                        l_rnfn = criterion(out_rnfn, targets[:, 2].unsqueeze(1))
                        loss = l_scs + l_lnfn + l_rnfn
                    acc_batch = compute_accuracy((out_scs, out_lnfn, out_rnfn), targets, classification_mode)
                val_run_loss += loss.item()
                val_corr += acc_batch * len(inputs)
                val_samples += len(inputs)

        avg_val_loss = val_run_loss / len(val_loader)
        avg_val_acc = val_corr / val_samples

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        logger.log(epoch + 1, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Train Acc={avg_train_acc:.4f}, Val Acc={avg_val_acc:.4f}")

        best_val_loss, patience_counter, stop_now = early_stopping_check(
            avg_val_loss, best_val_loss, patience_counter, early_stopping_patience
        )
        if patience_counter == 0:
            best_path = os.path.join(logger.get_log_dir(), "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Best model saved at epoch {epoch+1} => {best_path}")
        if stop_now:
            print(f"[INFO] Early stopping at epoch {epoch+1}.")
            break

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, figsize=(10, 5))
    print(f"[INFO] Training complete. Logs & model in {logger.get_log_dir()}")


if __name__ == "__main__":
    train_model()
