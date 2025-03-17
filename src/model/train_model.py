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

def load_config(config_path="config.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_depth_from_folder(folder_name: str):
    # Example: "target_window_128x128_5D_B2A2" -> extract 5 from "5D"
    parts = folder_name.split("_")
    for p in parts:
        if p.endswith("D") and p[:-1].isdigit():
            return int(p[:-1])
    return None

def parse_size_from_folder(folder_name: str):
    # Example: "target_window_128x128_5D_B2A2" -> extract (128, 128)
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
    # Use stratified splitting if possible.
    min_count = df[label_key].value_counts().min()
    if min_count < 2:
        train_df, temp_df = train_test_split(df, test_size=test_size, shuffle=True, random_state=seed)
    else:
        train_df, temp_df = train_test_split(df, test_size=test_size, stratify=df[label_key],
                                             shuffle=True, random_state=seed)
    if len(temp_df) == 0:
        return train_df, pd.DataFrame(), pd.DataFrame()
    min_count_temp = temp_df[label_key].value_counts().min()
    if min_count_temp < 2:
        val_df, test_df = train_test_split(temp_df, test_size=val_split, shuffle=True, random_state=seed)
    else:
        val_df, test_df = train_test_split(temp_df, test_size=val_split, stratify=temp_df[label_key],
                                           shuffle=True, random_state=seed)
    return train_df, val_df, test_df

def compute_loss(outputs, targets, classification_mode, criterion):
    if classification_mode == ClassificationMode.SINGLE_MULTICLASS:
        return criterion(outputs, targets)
    elif classification_mode == ClassificationMode.SINGLE_BINARY:
        return criterion(outputs, targets)
    elif classification_mode == ClassificationMode.MULTI_MULTICLASS:
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
    if classification_mode == ClassificationMode.SINGLE_MULTICLASS:
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        return correct / (len(targets) + 1e-8)
    elif classification_mode == ClassificationMode.SINGLE_BINARY:
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct = (preds == targets).sum().item()
        return correct / (targets.numel() + 1e-8)
    elif classification_mode == ClassificationMode.MULTI_MULTICLASS:
        out_scs, out_lnfn, out_rnfn = outputs
        bsz = targets.shape[0]
        scs_preds  = torch.argmax(out_scs, dim=1)
        lnfn_preds = torch.argmax(out_lnfn, dim=1)
        rnfn_preds = torch.argmax(out_rnfn, dim=1)
        total_correct = ((scs_preds == targets[:,0]).sum().item() +
                         (lnfn_preds == targets[:,1]).sum().item() +
                         (rnfn_preds == targets[:,2]).sum().item())
        return total_correct / (3 * bsz + 1e-8)
    else:  # MULTI_BINARY
        out_scs, out_lnfn, out_rnfn = outputs
        bsz = targets.shape[0]
        scs_preds = (torch.sigmoid(out_scs) > 0.5).float()
        lnfn_preds = (torch.sigmoid(out_lnfn) > 0.5).float()
        rnfn_preds = (torch.sigmoid(out_rnfn) > 0.5).float()
        total_correct = ((scs_preds.squeeze(1) == targets[:,0]).sum().item() +
                         (lnfn_preds.squeeze(1) == targets[:,1]).sum().item() +
                         (rnfn_preds.squeeze(1) == targets[:,2]).sum().item())
        return total_correct / (3 * bsz + 1e-8)

def focal_loss(inputs, targets, alpha=1.0, gamma=2.0, reduction='mean'):
    """
    Compute the focal loss for multi-class classification.

    Args:
        inputs (Tensor): Logits with shape [N, C].
        targets (Tensor): Ground truth labels with shape [N] (LongTensor).
        alpha (float or Tensor): Balancing factor. Can be a scalar or a tensor of shape [C].
        gamma (float): Focusing parameter.
        reduction (str): 'mean', 'sum', or 'none'.

    Returns:
        Tensor: Loss value.
    """
    ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # probability of correct classification

    if isinstance(alpha, torch.Tensor):
        alpha_t = alpha[targets]
    else:
        alpha_t = alpha

    loss = alpha_t * ((1 - pt) ** gamma) * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def build_disc_dirs_map(discs_for_training, subfolder, interim_base_path):
    """
    Build a dictionary mapping each disc label (without '/') to its folder.
    For example: {"L4L5": "./data/interim/L4L5/target_window_128x128_5D_B2A2", ...}
    """
    disc_dirs = {}
    for d in discs_for_training:
        key = d.replace("/", "")
        path = os.path.join(interim_base_path, key, subfolder)
        disc_dirs[key] = path
    return disc_dirs

def map_binary_label(label, mapping_mode, classification_mode):
    # For single binary classification:
    if classification_mode == ClassificationMode.SINGLE_BINARY:
        if mapping_mode == "severe_positive":
            # Treat labels 0, 1 as negative (0), others as positive (1)
            return 0 if label in [0, 1] else 1
        else:
            # Default: only label 0 is negative; all others become positive.
            return 0 if label == 0 else 1
    # For multi binary classification:
    elif classification_mode == ClassificationMode.MULTI_BINARY:
        if mapping_mode == "severe_positive":
            # For multi, you may want a different mapping; for example, labels 0 and 1 negative, others positive.
            return 0 if label in [0, 1] else 1
        else:
            return 0 if label == 0 else 1
    else:
        return label  # for multiclass modes, return the original label

def train_model(config_path="config.yml"):
    # --- 1. Load configuration and set up logger ---
    config = load_config(config_path)
    seed = config["project"]["seed"]
    base_logger = logging.getLogger(__name__)
    prefix = f"{config['training']['model_arch']}_{config['training']['classification_mode']}_{config['training']['selected_tensor_subfolder']}_{config['training']['batch_size']}_{config['training']['num_epochs']}_{config['training']['learning_rate']}_{config['training']['dropout_prob']}"
    logger = TrainingLogger(prefix)
    log_file_path = os.path.join(logger.get_log_dir(), "training.log")
    setup_python_logger(base_logger, log_file_path)

    base_logger.info("[INIT] Starting train_model...")
    for k, v in config["training"].items():
        base_logger.info(f"{k}: {v}")
    base_logger.info(f"Log Directory: {logger.get_log_dir()}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 2. Extract parameters ---
    classification_str = config["training"]["classification_mode"]
    classification_mode = ClassificationMode(classification_str)
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
    binary_mapping_mode = config["training"].get("binary_mapping_mode", "normal_negative")

    # --- 3. Read merged CSV, properly handle duplicates and filter by disc ---
    merged_csv_path = os.path.join(config["data"]["raw_path"], "merged_train_data.csv")
    df_all = pd.read_csv(merged_csv_path)

    # Add disc_label column (remove "/" from level)
    df_all["disc_label"] = df_all["level"].str.replace("/", "")

    # Filter to only rows with discs we want
    df_filtered = df_all[df_all["level"].isin(discs_for_training)].copy()

    # Exclude specified studies
    if excluded_studies:
        df_filtered = df_filtered[~df_filtered["study_id"].isin(excluded_studies)].copy()

    # Remove duplicates based on study_id and level
    df_filtered = df_filtered.drop_duplicates(subset=["study_id", "level"]).copy()

    if df_filtered.empty:
        base_logger.warning("No data found for requested discs. Exiting.")
        return

    base_logger.info(f"After filtering to discs {discs_for_training} and removing duplicates, shape={df_filtered.shape}")

    # --- 4. Drop rows with missing labels and split data ---
    if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
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
    base_logger.info(f"Final shape after dropping NA labels: {df_filtered.shape}")

    train_df, val_df, test_df = create_data_splits(df_filtered, final_label_key, test_size, val_split, seed)
    base_logger.info(f"Train size={len(train_df)}, Val size={len(val_df)}, Test size={len(test_df)}")

    # --- Log overall and per-disc distribution using binary mapping if in binary mode ---
    if classification_mode in [ClassificationMode.SINGLE_BINARY, ClassificationMode.MULTI_BINARY]:
        base_logger.info("Overall binary training dataset distribution:")
        binary_overall = train_df[final_label_key].apply(
            lambda x: map_binary_label(x, binary_mapping_mode, classification_mode)
        ).value_counts().to_dict()
        for cls, cnt in binary_overall.items():
            base_logger.info(f"  Class {cls}: {cnt}")

        base_logger.info("Per-disc binary training dataset distribution:")
        for disc in discs_for_training:
            disc_key = disc.replace("/", "")
            disc_df = train_df[train_df["disc_label"] == disc_key]
            binary_disc = disc_df[final_label_key].apply(
                lambda x: map_binary_label(x, binary_mapping_mode, classification_mode)
            ).value_counts().to_dict()
            base_logger.info(f"  For disc {disc} (processed as '{disc_key}'):")
            for cls, cnt in binary_disc.items():
                base_logger.info(f"    Class {cls}: {cnt}")
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

    if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        in_channels = 1
    else:
        in_channels = 3

    # --- 4.5. Handle imbalance methods ---
    imbalance_cfg = config["training"].get("imbalance", {})
    use_class_weights = imbalance_cfg.get("use_class_weights", False)
    use_weighted_sampler = imbalance_cfg.get("use_weighted_sampler", False)
    loss_type = imbalance_cfg.get("loss_type", "cross_entropy")  # options: "cross_entropy", "focal"
    
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

    # --- 5. Build disc_dirs and dataset objects ---
    disc_dirs = build_disc_dirs_map(discs_for_training, subfolder, interim_base_path)
    if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        train_ds = SingleDiseaseDatasetMultiDisc(
            train_df, disc_dirs, disease,
            parse_depth_from_folder(subfolder),
            parse_size_from_folder(subfolder),
            classification_mode=classification_str
        )
        val_ds = SingleDiseaseDatasetMultiDisc(
            val_df, disc_dirs, disease,
            parse_depth_from_folder(subfolder),
            parse_size_from_folder(subfolder),
            classification_mode=classification_str
        )
        test_ds = SingleDiseaseDatasetMultiDisc(
            test_df, disc_dirs, disease,
            parse_depth_from_folder(subfolder),
            parse_size_from_folder(subfolder),
            classification_mode=classification_str
        )
    else:
        train_ds = MultiLabelSpinalDatasetMultiDisc(
            train_df, disc_dirs,
            parse_depth_from_folder(subfolder),
            parse_size_from_folder(subfolder),
            classification_mode=classification_str
        )
        val_ds = MultiLabelSpinalDatasetMultiDisc(
            val_df, disc_dirs,
            parse_depth_from_folder(subfolder),
            parse_size_from_folder(subfolder),
            classification_mode=classification_str
        )
        test_ds = MultiLabelSpinalDatasetMultiDisc(
            test_df, disc_dirs,
            parse_depth_from_folder(subfolder),
            parse_size_from_folder(subfolder),
            classification_mode=classification_str
        )

    if use_weighted_sampler and sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  collate_fn=custom_collate_filter_none)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=custom_collate_filter_none)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=custom_collate_filter_none)

    # Count and log skipped samples (if tensor files missing or shapes mismatch)
    skipped = sum(1 for i in range(len(train_ds)) if train_ds[i] is None or train_ds[i][0] is None)
    base_logger.info(f"Skipped (training) samples: {skipped} out of {len(train_ds)}")

    # --- 6. Build model, optimizer, and (optionally) scheduler ---
    model, criterion = build_model(config, in_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Override criterion if using class weights
    if classification_mode == ClassificationMode.SINGLE_BINARY and use_class_weights:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    elif classification_mode == ClassificationMode.SINGLE_MULTICLASS and use_class_weights and loss_type == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    scheduler = None
    if config["training"].get("use_lr_scheduler", False):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config["training"].get("lr_scheduler_factor", 0.1),
            patience=config["training"].get("lr_scheduler_patience", 10),
            verbose=True
        )

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

        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}"):
            if batch is None:
                continue
            x, y, _ = batch  # disc_label not used in training
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            # Use focal loss if specified
            if loss_type == "focal" and classification_mode == ClassificationMode.SINGLE_MULTICLASS:
                focal_params = imbalance_cfg.get("focal", {})
                alpha_val = focal_params.get("alpha", 1.0)
                if isinstance(alpha_val, list):
                    alpha_val = torch.tensor(alpha_val, dtype=torch.float, device=device)
                loss = focal_loss(outputs, y, alpha=alpha_val, gamma=focal_params.get("gamma", 2.0))
            else:
                loss = compute_loss(outputs, y, classification_mode, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_acc = compute_accuracy(outputs, y, classification_mode)
            total_samples += len(x)
            running_correct += batch_acc * len(x)

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = running_correct / (total_samples + 1e-8)

        # Validation
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
                loss_val = compute_loss(outputs_val, y_val, classification_mode, criterion)
                val_loss_sum += loss_val.item()
                batch_val_acc = compute_accuracy(outputs_val, y_val, classification_mode)
                val_samples += len(x_val)
                val_correct += batch_val_acc * len(x_val)

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_acc = val_correct / (val_samples + 1e-8)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        logger.log(epoch+1, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc)
        base_logger.info(f"[Epoch {epoch+1}/{num_epochs}] Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Train Acc={avg_train_acc:.4f}, Val Acc={avg_val_acc:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

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

    base_logger.info(f"[INFO] Training complete. Logs & models saved in {logger.get_log_dir()}")
    # Optionally plot curves:
    # plot_training_curves(train_losses, val_losses, train_accs, val_accs)

if __name__ == "__main__":
    train_model("config.yml")
