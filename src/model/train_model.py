import os
import yaml
import torch
import numpy as np
import pandas as pd
import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from collections import Counter

from src.utils.enums import ClassificationMode
from src.model.dataset import (
    SingleDiseaseDatasetMultiDisc,
    MultiLabelSpinalDatasetMultiDisc,
    custom_collate_filter_none
)
from src.model.model_builder import build_model
from src.utils.logger import TrainingLogger, setup_python_logger
from src.visualization.visualize import plot_training_curves
from src.model.imbalance_utils import apply_imbalance_strategy


def load_config(config_path="config.yml"):
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_depth_from_folder(folder_name: str):
    """
    Extract the depth (number of slices) from a folder name.

    For example, for the folder name "target_window_128x128_5D_B2A2", it returns 5.

    Args:
        folder_name (str): Name of the folder containing information on depth.

    Returns:
        int or None: Extracted depth as an integer, or None if not found.
    """
    parts = folder_name.split("_")
    for p in parts:
        if p.endswith("D") and p[:-1].isdigit():
            return int(p[:-1])
    return None


def parse_size_from_folder(folder_name: str):
    """
    Extract the spatial size (width, height) from a folder name.

    For example, for the folder name "target_window_128x128_5D_B2A2", it returns (128, 128).

    Args:
        folder_name (str): Folder name containing size information.

    Returns:
        tuple or None: (width, height) as integers, or None if extraction fails.
    """
    parts = folder_name.split("_")
    for p in parts:
        if "x" in p:
            try:
                w, h = p.split("x")
                return (int(w), int(h))
            except:
                continue
    return None


def create_data_splits(df, label_key, test_size, val_split, seed):
    """
    Split the dataset into training, validation, and test sets.
    
    If the smallest class count is at least 2, a stratified split is performed.

    Args:
        df (pd.DataFrame): Input dataframe.
        label_key (str): Column name to use for stratification (if possible).
        test_size (float): Proportion of the dataset to include in the test split.
        val_split (float): Proportion of the temporary split to allocate for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_df, val_df, test_df) dataframes.
    """
    # Determine whether stratification is possible by checking the minimum count of labels.
    min_count = df[label_key].value_counts().min()
    if min_count < 2:
        train_df, temp_df = train_test_split(
            df, test_size=test_size, shuffle=True, random_state=seed
        )
    else:
        train_df, temp_df = train_test_split(
            df, test_size=test_size, stratify=df[label_key],
            shuffle=True, random_state=seed
        )

    # If temporary split is empty, return empty dataframes for validation and test.
    if len(temp_df) == 0:
        return train_df, pd.DataFrame(), pd.DataFrame()

    min_count_temp = temp_df[label_key].value_counts().min()
    if min_count_temp < 2:
        val_df, test_df = train_test_split(temp_df, test_size=val_split, shuffle=True,
                                           random_state=seed)
    else:
        val_df, test_df = train_test_split(temp_df, test_size=val_split,
                                           stratify=temp_df[label_key],
                                           shuffle=True, random_state=seed)
    return train_df, val_df, test_df


def compute_loss(outputs, targets, classification_mode, criterion):
    """
    Compute the classification loss based on the classification mode.

    For multi-label/multi-task problems, the loss is computed for each output branch
    and then summed.

    Args:
        outputs: Model outputs. For multi-task, expected to be a tuple of outputs.
        targets: Ground truth labels.
        classification_mode: An instance of ClassificationMode.
        criterion: Loss function (e.g., CrossEntropyLoss or BCEWithLogitsLoss).

    Returns:
        torch.Tensor: The computed loss value.
    """
    mode = classification_mode
    if mode == ClassificationMode.SINGLE_MULTICLASS:
        return criterion(outputs, targets)
    elif mode == ClassificationMode.SINGLE_BINARY:
        return criterion(outputs, targets)
    elif mode == ClassificationMode.MULTI_MULTICLASS:
        out_scs, out_lnfn, out_rnfn = outputs
        loss = (criterion(out_scs, targets[:, 0]) +
                criterion(out_lnfn, targets[:, 1]) +
                criterion(out_rnfn, targets[:, 2]))
        return loss
    else:  # MULTI_BINARY
        out_scs, out_lnfn, out_rnfn = outputs
        loss = (criterion(out_scs, targets[:, 0].unsqueeze(1)) +
                criterion(out_lnfn, targets[:, 1].unsqueeze(1)) +
                criterion(out_rnfn, targets[:, 2].unsqueeze(1)))
        return loss


def compute_accuracy(outputs, targets, classification_mode):
    """
    Compute the accuracy (or average accuracy for multi-task) for a given batch.

    Args:
        outputs: Model outputs.
        targets: Ground truth labels.
        classification_mode: An instance of ClassificationMode indicating the task type.

    Returns:
        float: The computed accuracy.
    """
    mode = classification_mode
    if mode == ClassificationMode.SINGLE_MULTICLASS:
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        return correct / (len(targets) + 1e-8)
    elif mode == ClassificationMode.SINGLE_BINARY:
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct = (preds == targets).sum().item()
        return correct / (targets.numel() + 1e-8)
    elif mode == ClassificationMode.MULTI_MULTICLASS:
        out_scs, out_lnfn, out_rnfn = outputs
        bsz = targets.shape[0]
        scs_preds  = torch.argmax(out_scs, dim=1)
        lnfn_preds = torch.argmax(out_lnfn, dim=1)
        rnfn_preds = torch.argmax(out_rnfn, dim=1)
        total_correct = ((scs_preds == targets[:, 0]).sum().item() +
                         (lnfn_preds == targets[:, 1]).sum().item() +
                         (rnfn_preds == targets[:, 2]).sum().item())
        return total_correct / (3 * bsz + 1e-8)
    else:  # MULTI_BINARY
        out_scs, out_lnfn, out_rnfn = outputs
        bsz = targets.shape[0]
        scs_preds = (torch.sigmoid(out_scs) > 0.5).float()
        lnfn_preds = (torch.sigmoid(out_lnfn) > 0.5).float()
        rnfn_preds = (torch.sigmoid(out_rnfn) > 0.5).float()
        total_correct = ((scs_preds.squeeze(1) == targets[:, 0]).sum().item() +
                         (lnfn_preds.squeeze(1) == targets[:, 1]).sum().item() +
                         (rnfn_preds.squeeze(1) == targets[:, 2]).sum().item())
        return total_correct / (3 * bsz + 1e-8)


def map_label_3class_to_2class(label, mode="normal_negative"):
    """
    For single-binary classification:
    Convert a 3-class label into a binary label.
    
    "severe_positive": maps label 2 to 1 and others to 0.
    "normal_negative": maps label 0 to 0 and labels 1 or 2 to 1.
    
    Args:
        label (int): Original label.
        mode (str): Mapping mode ("normal_negative" or "severe_positive").

    Returns:
        int: The binary-mapped label.
    """
    if mode == "severe_positive":
        return 1 if label == 2 else 0
    else:
        return 0 if label == 0 else 1


def build_disc_dirs_map(discs_for_training, subfolder, interim_base_path):
    """
    Create a dictionary mapping each disc (with slashes removed) to its directory path.
    
    Args:
        discs_for_training (list of str): List of disc identifiers (e.g., "L4/L5").
        subfolder (str): Subfolder name where tensor volumes are stored.
        interim_base_path (str): Base directory for interim data.
    
    Returns:
        dict: Mapping from disc key (e.g., "L4L5") to full path.
    """
    disc_dirs = {}
    for d in discs_for_training:
        key = d.replace("/", "")
        path = os.path.join(interim_base_path, key, subfolder)
        disc_dirs[key] = path
    return disc_dirs


def train_model(config_path="config.yml"):
    """
    Main training pipeline for the lumbar spine degenerative classification model.
    
    This function loads the configuration, sets up logging and seeds, loads and filters the data,
    applies any necessary data balancing strategies, splits the data into train/validation/test sets,
    builds the dataset and DataLoader objects, constructs the model and optimizer, and then runs
    the training loop with optional early stopping and learning rate scheduling. Finally, it saves
    the data splits and the best model checkpoints.
    
    Args:
        config_path (str): Path to the configuration YAML file (default "config.yml").

    Returns:
        None
    """
    # --- 1. Load configuration and set seed ---
    config = load_config(config_path)
    seed = config["project"]["seed"]

    # Setup logging using a custom TrainingLogger.
    base_logger = logging.getLogger(__name__)
    prefix = (f"{config['training']['model_arch']}_"
              f"{config['training']['classification_mode']}_"
              f"{config['training']['selected_tensor_subfolder']}_"
              f"{config['training']['batch_size']}_"
              f"{config['training']['num_epochs']}_"
              f"{config['training']['learning_rate']}_"
              f"{config['training']['dropout_prob']}")
    logger = TrainingLogger(prefix)
    log_file_path = os.path.join(logger.get_log_dir(), "training.log")
    setup_python_logger(base_logger, log_file_path)

    base_logger.info("[INIT] Starting train_model...")
    for k, v in config["training"].items():
        base_logger.info(f"{k}: {v}")
    base_logger.info(f"Log Directory: {logger.get_log_dir()}")

    # Set random seeds for reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 2. Extract training parameters ---
    classification_str = config["training"]["classification_mode"]
    classification_mode = ClassificationMode(classification_str)
    binary_mapping_mode = config["training"].get("binary_mapping_mode", "normal_negative")

    disease = config["training"]["disease"]
    test_size = config["training"]["test_size"]
    val_split = config["training"]["validation_split_of_temp"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    lr = config["training"]["learning_rate"]
    early_stop = config["training"]["early_stopping_patience"]
    dropout_prob = config["training"]["dropout_prob"]
    excluded_studies = config["training"].get("excluded_studies", [])
    discs_for_training = config["training"].get("discs_for_training", ["L4/L5"])
    subfolder = config["training"]["selected_tensor_subfolder"]
    interim_base_path = config["data"]["interim_base_path"]

    # Extract imbalance configuration.
    imbalance_cfg = config["training"].get("imbalance", {})
    method = imbalance_cfg.get("method", "none")
    undersampling_ratio = imbalance_cfg.get("undersampling_ratio", 0.5)
    oversampling_ratio = imbalance_cfg.get("oversampling_ratio", 1.0)
    smote = imbalance_cfg.get("smote", False)

    # --- 3. Load and filter the merged CSV data ---
    merged_csv_path = os.path.join(config["data"]["raw_path"], "merged_train_data.csv")
    df_all = pd.read_csv(merged_csv_path)

    # Filter the data to include only the specified training discs.
    df_all["disc_label"] = df_all["level"].str.replace("/", "")
    df_filtered = df_all[df_all["level"].isin(discs_for_training)].copy()

    # Remove duplicate samples.
    df_filtered = df_filtered.drop_duplicates(subset=["study_id", "level"]).copy()
    base_logger.info(f"After filtering to discs {discs_for_training} and removing duplicates, shape={df_filtered.shape}")

    # Select and prepare the target labels.
    if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        label_col = f"{disease}_label"
        df_filtered = df_filtered.dropna(subset=[label_col])
        df_filtered[label_col] = df_filtered[label_col].astype(int)

        # For binary classification, convert 3-class labels to binary using the mapping mode.
        if classification_mode == ClassificationMode.SINGLE_BINARY:
            df_filtered[label_col] = df_filtered[label_col].apply(
                lambda x: map_label_3class_to_2class(x, mode=binary_mapping_mode)
            )
        final_label_key = label_col
    else:
        # For multi-label tasks, combine multiple label columns.
        needed_cols = ["scs_label", "lnfn_label", "rnfn_label"]
        df_filtered = df_filtered.dropna(subset=needed_cols)
        for c in needed_cols:
            df_filtered[c] = df_filtered[c].astype(int)
        df_filtered["multi_label"] = (df_filtered["scs_label"].astype(str) + "_" +
                                      df_filtered["lnfn_label"].astype(str) + "_" +
                                      df_filtered["rnfn_label"].astype(str))
        final_label_key = "multi_label"

    base_logger.info(f"Final shape after dropping NA labels: {df_filtered.shape}")

    # --- 4. Create data splits (train/val/test) and persist them ---
    train_df, val_df, test_df = create_data_splits(df_filtered, final_label_key, test_size, val_split, seed)
    base_logger.info(f"Train size={len(train_df)}, Val size={len(val_df)}, Test size={len(test_df)}")

    # Apply the imbalance strategy to the training set (if specified).
    if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        base_logger.info("Overall binary training dataset distribution:")
        binary_overall = train_df[final_label_key].value_counts().to_dict()
        for cls, cnt in binary_overall.items():
            base_logger.info(f"  Class {cls}: {cnt}")

        base_logger.info("Per-disc binary training dataset distribution:")
        for disc in discs_for_training:
            disc_key = disc.replace("/", "")
            disc_df = train_df[train_df["disc_label"] == disc_key]
            binary_disc = disc_df[final_label_key].value_counts().to_dict()
            base_logger.info(f"  For disc {disc} (processed as '{disc_key}'):")
            for cls, cnt in binary_disc.items():
                base_logger.info(f"    Class {cls}: {cnt}")
        balanced_train_df = apply_imbalance_strategy(
            df=train_df,
            label_col=final_label_key,
            method=method,
            undersampling_ratio=undersampling_ratio,
            oversampling_ratio=oversampling_ratio,
            smote=smote,
            is_multilabel=False  # single-label scenario
        )
    else:
        base_logger.info("Overall training dataset distribution:")
        overall_counts = train_df[final_label_key].value_counts().to_dict()
        for cls, cnt in overall_counts.items():
            base_logger.info(f"  Class {cls}: {cnt}")
        base_logger.info("Per-disc training dataset distribution:")
        for disc in discs_for_training:
            disc_key = disc.replace("/", "")
            disc_df = train_df[train_df["disc_label"] == disc_key]
            disc_counts = disc_df[final_label_key].value_counts().to_dict()
            base_logger.info(f"  For disc {disc} (processed as '{disc_key}'):")
            for cls, cnt in disc_counts.items():
                base_logger.info(f"    Class {cls}: {cnt}")
        balanced_train_df = apply_imbalance_strategy(
            df=train_df,
            label_col="multi_label",
            method=method,
            undersampling_ratio=undersampling_ratio,
            oversampling_ratio=oversampling_ratio,
            smote=smote,
            is_multilabel=True
        )
    train_df = balanced_train_df

    if method != "none":
        base_logger.info(f"[AFTER] Train size={len(train_df)}, Val size={len(val_df)}, Test size={len(test_df)}")
        if classification_mode in [ClassificationMode.SINGLE_BINARY, ClassificationMode.MULTI_BINARY]:
            base_logger.info("Binary distribution in TRAIN after re-sample:")
            binary_overall = train_df[final_label_key].value_counts().to_dict()
            for cls, cnt in binary_overall.items():
                base_logger.info(f"  Class {cls}: {cnt}")
            base_logger.info("Per-disc binary training dataset distribution:")
            for disc in discs_for_training:
                disc_key = disc.replace("/", "")
                disc_df = train_df[train_df["disc_label"] == disc_key]
                binary_disc = disc_df[final_label_key].value_counts().to_dict()
                base_logger.info(f"  For disc {disc} (processed as '{disc_key}'):")
                for cls, cnt in binary_disc.items():
                    base_logger.info(f"    Class {cls}: {cnt}")
        else:
            base_logger.info("Distribution in TRAIN after re-sample:")
            overall_counts = train_df[final_label_key].value_counts().to_dict()
            for cls, cnt in overall_counts.items():
                base_logger.info(f"  Class {cls}: {cnt}")
            base_logger.info("Per-disc training dataset distribution:")
            for disc in discs_for_training:
                disc_key = disc.replace("/", "")
                disc_df = train_df[train_df["disc_label"] == disc_key]
                disc_counts = disc_df[final_label_key].value_counts().to_dict()
                base_logger.info(f"  For disc {disc} (processed as '{disc_key}'):")
                for cls, cnt in disc_counts.items():
                    base_logger.info(f"    Class {cls}: {cnt}")

    # Persist the final data splits in the model's log directory.
    split_dir = logger.get_log_dir()  # e.g., ./models/<timestamp>
    train_split_path = os.path.join(split_dir, "train_split.csv")
    val_split_path = os.path.join(split_dir, "val_split.csv")
    test_split_path = os.path.join(split_dir, "test_split.csv")

    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)
    test_df.to_csv(test_split_path, index=False)
    base_logger.info(f"Saved splits to: {split_dir}")

    # --- 5. Build dataset objects for training and validation ---
    use_class_weights = imbalance_cfg.get("use_class_weights", False)
    use_weighted_sampler = imbalance_cfg.get("use_weighted_sampler", False)

    class_weights = None
    sample_weights = None
    if use_class_weights:
        if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
            labels = train_df[label_col].tolist()
            counts = Counter(labels)
            total = len(labels)
            if classification_mode == ClassificationMode.SINGLE_BINARY:
                num_0 = sum(1 for x in labels if x == 0)
                num_1 = sum(1 for x in labels if x != 0)
                pos_weight = (num_0 / num_1) if num_1 > 0 else 1.0
                base_logger.info(f"Computed pos_weight for BCEWithLogitsLoss: {pos_weight:.3f}")
            else:  # SINGLE_MULTICLASS
                sorted_keys = sorted(counts.keys())
                weights = [total / counts[k] for k in sorted_keys]
                class_weights = torch.tensor(weights, dtype=torch.float)
                base_logger.info(f"Computed class weights for CrossEntropyLoss: {class_weights}")
        else:  # Multi-label: compute sample weights
            labels = train_df["multi_label"].tolist()
            counts = Counter(labels)
            total = len(labels)
            mapping = {label: total / count for label, count in counts.items()}
            base_logger.info(f"Computed sample weight mapping for multi-label: {mapping}")

    if use_weighted_sampler:
        if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
            labels = train_df[label_col].tolist()
            counts = Counter(labels)
            total = len(labels)
            mapping = {label: total / counts[label] for label in counts}
            sample_weights = [mapping[label] for label in labels]
        else:
            labels = train_df["multi_label"].tolist()
            counts = Counter(labels)
            total = len(labels)
            mapping = {label: total / counts[label] for label in counts}
            sample_weights = [mapping[label] for label in labels]
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
        base_logger.info("Using WeightedRandomSampler for training DataLoader.")

    # Create dataset objects for training and validation.
    if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        in_channels = 1
        train_ds = SingleDiseaseDatasetMultiDisc(
            train_df,
            build_disc_dirs_map(discs_for_training, subfolder, interim_base_path),
            disease=disease,
            final_depth=parse_depth_from_folder(subfolder),
            final_size=parse_size_from_folder(subfolder),
            classification_mode=classification_str
        )
        val_ds = SingleDiseaseDatasetMultiDisc(
            val_df,
            build_disc_dirs_map(discs_for_training, subfolder, interim_base_path),
            disease=disease,
            final_depth=parse_depth_from_folder(subfolder),
            final_size=parse_size_from_folder(subfolder),
            classification_mode=classification_str
        )
    else:
        in_channels = 3
        train_ds = MultiLabelSpinalDatasetMultiDisc(
            train_df,
            build_disc_dirs_map(discs_for_training, subfolder, interim_base_path),
            final_depth=parse_depth_from_folder(subfolder),
            final_size=parse_size_from_folder(subfolder),
            classification_mode=classification_str
        )
        val_ds = MultiLabelSpinalDatasetMultiDisc(
            val_df,
            build_disc_dirs_map(discs_for_training, subfolder, interim_base_path),
            final_depth=parse_depth_from_folder(subfolder),
            final_size=parse_size_from_folder(subfolder),
            classification_mode=classification_str
        )

    # Create DataLoader objects.
    if use_weighted_sampler:
        from torch.utils.data import WeightedRandomSampler
        labels_list = train_df[final_label_key].tolist()
        c = Counter(labels_list)
        total = len(labels_list)
        mapping = {lab: total / c[lab] for lab in c}
        sample_weights = [mapping[lab] for lab in labels_list]
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights_tensor,
                                        num_samples=len(sample_weights_tensor),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  collate_fn=custom_collate_filter_none)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=custom_collate_filter_none)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=custom_collate_filter_none)

    # Log number of skipped samples (if any) in the training dataset.
    skipped = sum(1 for i in range(len(train_ds)) if train_ds[i] is None or train_ds[i][0] is None)
    base_logger.info(f"Skipped (training) samples: {skipped} out of {len(train_ds)}")

    # --- 6. Build model & optimizer ---
    model, criterion = build_model(config, in_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Optional learning rate scheduler.
    scheduler = None
    if config["training"].get("use_lr_scheduler", False):
        factor_ = config["training"].get("lr_scheduler_factor", 0.1)
        pat_ = config["training"].get("lr_scheduler_patience", 10)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor_, patience=pat_, verbose=True)

    # --- 7. Training loop ---
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_correct = 0.0, 0.0
        total_samples = 0

        # Iterate over training batches.
        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}"):
            if batch is None:
                continue
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)

            # Compute loss (focal loss logic can be added here if required).
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_acc = compute_accuracy(outputs, y, classification_mode)
            total_samples += len(x)
            running_correct += batch_acc * len(x)

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = running_correct / (total_samples + 1e-8)

        # Validation loop.
        model.eval()
        val_loss_sum, val_correct = 0.0, 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}/{num_epochs}"):
                if batch is None:
                    continue
                x_val, y_val, _ = batch
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs_val = model(x_val)
                val_loss = criterion(outputs_val, y_val)
                val_loss_sum += val_loss.item()
                batch_val_acc = compute_accuracy(outputs_val, y_val, classification_mode)
                val_samples += len(x_val)
                val_correct += batch_val_acc * len(x_val)

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_acc = val_correct / (val_samples + 1e-8)

        # Record losses and accuracies.
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        # Log current epoch metrics.
        logger.log(epoch+1, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc)
        base_logger.info(f"[Epoch {epoch+1}/{num_epochs}] Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Train Acc={avg_train_acc:.4f}, Val Acc={avg_val_acc:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

        # Save model checkpoints.
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
            best_path = os.path.join(logger.get_log_dir(), "best_model.pth")
            torch.save(model.state_dict(), best_path)
            base_logger.info(f"[Checkpoint] New best val acc: {best_val_acc:.4f}, model saved -> {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                base_logger.info(f"[INFO] Early stopping at epoch {epoch+1}")
                break

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_loss_path = os.path.join(logger.get_log_dir(), "best_val_loss_model.pth")
            torch.save(model.state_dict(), best_loss_path)
            base_logger.info(f"[Checkpoint] New best val loss: {best_val_loss:.4f}, model saved -> {best_loss_path}")

    base_logger.info(f"[INFO] Training complete. Logs & splits & models in {logger.get_log_dir()}")
    # Optionally, plot training curves.
    # plot_training_curves(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    train_model("config.yml")
