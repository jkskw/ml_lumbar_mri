import os
import yaml
import torch
import numpy as np
import pandas as pd
import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.model_selection import train_test_split

from src.utils.enums import ClassificationMode
from src.model.dataset import SingleDiseaseDataset, MultiLabelSpinalDataset
from src.model.model_builder import build_model
from src.utils.logger import TrainingLogger, setup_python_logger
from src.visualization.visualize import plot_training_curves


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_depth_from_folder(folder_name: str):
    """
    Attempt to parse something like '5D' in the folder_name.
    E.g. 'target_window_128x128_5D_B2A2' => parse '5D' => 5
    """
    parts = folder_name.split("_")
    if len(parts) >= 4 and parts[3].endswith("D"):
        depth_str = parts[3]  # e.g. '5D'
        return int(depth_str.replace("D", ""))
    return None


def parse_size_from_folder(folder_name: str):
    """
    Attempt to parse the resolution from something like '128x128' in the folder name.
    E.g. 'target_window_128x128_5D_B2A2' => parse '128x128' => (128, 128)
    """
    parts = folder_name.split("_")
    if len(parts) >= 3:
        resolution_str = parts[2]
        if 'x' in resolution_str:
            w, h = resolution_str.split('x')
            return (int(w), int(h))
    return None


def create_data_splits(df, label_key, test_size, val_split, seed):
    """
    Splits df into (train, val, test) in a possibly stratified way if feasible.
    """
    min_count = df[label_key].value_counts().min()
    if min_count < 2:
        # Not enough examples for stratified
        train_df, temp_df = train_test_split(
            df, test_size=test_size, shuffle=True, random_state=seed
        )
    else:
        train_df, temp_df = train_test_split(
            df, test_size=test_size, stratify=df[label_key], shuffle=True, random_state=seed
        )
    # Next split the 'temp_df' into val/test
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


def compute_loss(outputs, targets, classification_mode, criterion):
    """
    Unifies single vs multi loss computation.
    """
    if classification_mode == ClassificationMode.SINGLE_MULTICLASS:
        # outputs shape: [B, 3], targets shape: [B]
        loss = criterion(outputs, targets)
    elif classification_mode == ClassificationMode.SINGLE_BINARY:
        # outputs shape: [B, 1], targets shape: [B, 1]
        loss = criterion(outputs, targets)
    elif classification_mode == ClassificationMode.MULTI_MULTICLASS:
        # outputs is a tuple of (out_scs, out_lnfn, out_rnfn), each [B, 3]
        out_scs, out_lnfn, out_rnfn = outputs
        # targets shape: [B, 3], with integer labels
        loss_scs = criterion(out_scs, targets[:, 0])
        loss_lnfn = criterion(out_lnfn, targets[:, 1])
        loss_rnfn = criterion(out_rnfn, targets[:, 2])
        loss = loss_scs + loss_lnfn + loss_rnfn
    else:  # MULTI_BINARY
        # outputs is (out_scs, out_lnfn, out_rnfn), each [B,1]
        out_scs, out_lnfn, out_rnfn = outputs
        loss_scs = criterion(out_scs, targets[:, 0].unsqueeze(1))
        loss_lnfn = criterion(out_lnfn, targets[:, 1].unsqueeze(1))
        loss_rnfn = criterion(out_rnfn, targets[:, 2].unsqueeze(1))
        loss = loss_scs + loss_lnfn + loss_rnfn

    return loss


def compute_accuracy(outputs, targets, classification_mode):
    """
    Unified accuracy computation for single vs multi classification.
    """
    import torch

    if classification_mode == ClassificationMode.SINGLE_MULTICLASS:
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        total = len(targets)
        return correct / (total + 1e-8)

    elif classification_mode == ClassificationMode.SINGLE_BINARY:
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct = (preds == targets).sum().item()
        total = targets.numel()
        return correct / (total + 1e-8)

    elif classification_mode == ClassificationMode.MULTI_MULTICLASS:
        out_scs, out_lnfn, out_rnfn = outputs
        bsz = targets.shape[0]

        scs_preds = torch.argmax(out_scs, dim=1)
        lnfn_preds = torch.argmax(out_lnfn, dim=1)
        rnfn_preds = torch.argmax(out_rnfn, dim=1)

        scs_corr = (scs_preds == targets[:, 0]).sum().item()
        lnfn_corr = (lnfn_preds == targets[:, 1]).sum().item()
        rnfn_corr = (rnfn_preds == targets[:, 2]).sum().item()
        return (scs_corr + lnfn_corr + rnfn_corr) / (3 * bsz + 1e-8)

    else:  # MULTI_BINARY
        out_scs, out_lnfn, out_rnfn = outputs
        bsz = targets.shape[0]

        scs_preds = (torch.sigmoid(out_scs) > 0.5).float()
        lnfn_preds = (torch.sigmoid(out_lnfn) > 0.5).float()
        rnfn_preds = (torch.sigmoid(out_rnfn) > 0.5).float()

        scs_corr = (scs_preds.squeeze(1) == targets[:, 0]).sum().item()
        lnfn_corr = (lnfn_preds.squeeze(1) == targets[:, 1]).sum().item()
        rnfn_corr = (rnfn_preds.squeeze(1) == targets[:, 2]).sum().item()
        return (scs_corr + lnfn_corr + rnfn_corr) / (3 * bsz + 1e-8)


def create_dataset(df: pd.DataFrame, classification_mode: ClassificationMode, config: dict):
    """
    Creates a dataset object (SingleDiseaseDataset or MultiLabelSpinalDataset)
    depending on classification_mode.
    """
    folder = config["training"]["selected_tensor_folder"]
    final_depth = parse_depth_from_folder(folder)
    final_size  = parse_size_from_folder(folder)
    disease = config["training"]["disease"]

    base_path = os.path.join(config["data"]["interim_path"], folder)
    if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        # Single disease
        tensor_dir = os.path.join(base_path, disease)
        dataset = SingleDiseaseDataset(
            dataframe=df,
            disease=disease,
            tensor_dir=tensor_dir,
            final_depth=final_depth,
            final_size=final_size,
            classification_mode=classification_mode.value  # the Dataset expects a string
        )
    else:
        # Multi disease
        scs_dir  = os.path.join(base_path, "scs")
        lnfn_dir = os.path.join(base_path, "lnfn")
        rnfn_dir = os.path.join(base_path, "rnfn")
        dataset = MultiLabelSpinalDataset(
            dataframe=df,
            scs_dir=scs_dir,
            lnfn_dir=lnfn_dir,
            rnfn_dir=rnfn_dir,
            final_depth=final_depth,
            final_size=final_size,
            classification_mode=classification_mode.value
        )
    return dataset


def train_model(config_path="config.yml"):
    # 1) Load config
    config = load_config(config_path)
    seed = config["project"]["seed"]

    # 2) Setup logging (both console and a file that goes into the model folder)
    #    We won't know the exact model folder name until we create the TrainingLogger. 
    #    So we temporarily log to console, then reconfigure after we have the model dir.
    base_logger = logging.getLogger(__name__)
    base_logger.setLevel(logging.INFO)
    base_logger.info("[INIT] Starting train_model...")

    # 3) Extract parameters from config
    classification_str = config["training"]["classification_mode"]  # e.g. "multi_binary"
    classification_mode = ClassificationMode(classification_str)

    disease = config["training"]["disease"]
    folder = config["training"]["selected_tensor_folder"]
    test_size = config["training"]["test_size"]
    val_split = config["training"]["validation_split_of_temp"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    lr = config["training"]["learning_rate"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    dropout_prob = config["training"]["dropout_prob"]
    excluded_studies = config["training"].get("excluded_studies", [])

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 4) Load the main CSV and filter
    raw_csv = os.path.join(config["data"]["raw_path"], "merged_train_data.csv")
    df = pd.read_csv(raw_csv)
    # Deduplicate
    df_dedup = df.groupby("study_id", as_index=False).first()

    # Exclude problem studies if any
    if excluded_studies:
        original_len = len(df_dedup)
        df_dedup = df_dedup[~df_dedup["study_id"].isin(excluded_studies)]
        base_logger.info(f"[INFO] Excluding {len(excluded_studies)} studies: {excluded_studies}")
        base_logger.info(f"[INFO] Data size from {original_len} -> {len(df_dedup)} after exclusion")

    # 5) Prepare data splits
    if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        label_col = f"{disease}_label"
        df_dedup = df_dedup.dropna(subset=[label_col])
        df_dedup[label_col] = df_dedup[label_col].astype(int)

        train_df, val_df, test_df = create_data_splits(
            df_dedup, label_col, test_size, val_split, seed
        )
        in_channels = 1

    else:
        needed_cols = ["scs_label", "lnfn_label", "rnfn_label"]
        df_dedup = df_dedup.dropna(subset=needed_cols)
        df_dedup[needed_cols] = df_dedup[needed_cols].astype(int)
        # create a combined label column for stratification
        df_dedup["multi_label"] = (
            df_dedup["scs_label"].astype(str) + "_" +
            df_dedup["lnfn_label"].astype(str) + "_" +
            df_dedup["rnfn_label"].astype(str)
        )
        train_df, val_df, test_df = create_data_splits(
            df_dedup, "multi_label", test_size, val_split, seed
        )
        in_channels = 3

    # 6) Build Datasets & Loaders
    train_ds = create_dataset(train_df, classification_mode, config)
    val_ds   = create_dataset(val_df,   classification_mode, config)
    test_ds  = create_dataset(test_df,  classification_mode, config)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # 7) Build model and criterion
    model, criterion = build_model(config, in_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 8) Create a TrainingLogger => folder inside ./models
    #    Then re-configure a file-based logger inside that folder.
    log_prefix = f"{config['training']['model_arch']}_{classification_str}_{folder}_{batch_size}_{num_epochs}_{lr}_{dropout_prob}"
    logger = TrainingLogger(log_prefix)
    # Now that we know the model folder, create a .log file there using `setup_python_logger`
    log_file_path = os.path.join(logger.get_log_dir(), "training.log")
    setup_python_logger(base_logger, log_file_path)

    base_logger.info(f"== Training with classification_mode={classification_str}, model_arch={config['training']['model_arch']} ==")
    base_logger.info(f"Train size={len(train_ds)}, Val size={len(val_ds)}, Test size={len(test_ds)}")
    base_logger.info(f"Device = {device}, LR={lr}, BatchSize={batch_size}, Epochs={num_epochs}, Dropout={dropout_prob}")

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # 9) Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_correct = 0.0, 0.0
        total_samples = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = compute_loss(outputs, targets, classification_mode, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_acc = compute_accuracy(outputs, targets, classification_mode)
            total_samples += len(inputs)
            running_correct += batch_acc * len(inputs)

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = running_correct / (total_samples + 1e-8)

        # Validation
        model.eval()
        val_running_loss, val_running_correct = 0.0, 0.0
        val_samples = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss = compute_loss(outputs, targets, classification_mode, criterion)
                val_running_loss += val_loss.item()

                val_batch_acc = compute_accuracy(outputs, targets, classification_mode)
                val_samples += len(inputs)
                val_running_correct += val_batch_acc * len(inputs)

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_acc = val_running_correct / (val_samples + 1e-8)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        # Log epoch results
        logger.log(epoch+1, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc)
        base_logger.info(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | "
            f"Train Acc={avg_train_acc:.4f} | Val Acc={avg_val_acc:.4f}"
        )

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_path = os.path.join(logger.get_log_dir(), "best_model.pth")
            torch.save(model.state_dict(), best_path)
            base_logger.info(f"  >> New best model saved at {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                base_logger.info(f"[INFO] Early stopping at epoch {epoch+1}. No improvement in {patience_counter} epochs.")
                break

    # 10) Plot curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, figsize=(10, 5))
    base_logger.info(f"[INFO] Training complete. Logs & model in {logger.get_log_dir()}")
