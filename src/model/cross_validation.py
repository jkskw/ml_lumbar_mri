import os
import yaml
import logging
import numpy as np
import pandas as pd
import copy  # For deepcopy-ing the model state_dict
from collections import Counter

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
from imblearn.metrics import classification_report_imbalanced

# ---------------------------------------------------------------------
# Reuse your existing modules (only those that won't conflict):
# ---------------------------------------------------------------------
from src.model.train_model import (
    load_config,
    create_data_splits,
    parse_depth_from_folder,
    parse_size_from_folder,
    map_label_3class_to_2class,
    compute_accuracy,           # optional if you want
    compute_balanced_accuracy   # optional if you want
)
from src.model.imbalance_utils import apply_imbalance_strategy
from src.model.dataset import (
    SingleDiseaseDatasetMultiDisc,
    MultiLabelSpinalDatasetMultiDisc,
    custom_collate_filter_none
)
from src.model.model_builder import build_model
from src.utils.enums import ClassificationMode
from src.utils.logger import setup_python_logger

# ---------------------------------------------------------------------
# 1) A small helper to create disc_dirs map
# ---------------------------------------------------------------------
def build_disc_dirs_map(discs_for_training, subfolder, interim_base_path):
    """
    Create a dictionary mapping each disc (with slashes removed) to its directory path.
    """
    disc_dirs = {}
    for d in discs_for_training:
        key = d.replace("/", "")
        path = os.path.join(interim_base_path, key, subfolder)
        disc_dirs[key] = path
    return disc_dirs

# ---------------------------------------------------------------------
# 2) Local `_to_predictions()` that handles single vs. multi properly
# ---------------------------------------------------------------------
def _to_predictions(outputs, classification_mode: ClassificationMode):
    """
    Convert raw model outputs into predicted class labels or multi-label predictions,
    depending on classification_mode.
    """
    if classification_mode == ClassificationMode.SINGLE_MULTICLASS:
        return torch.argmax(outputs, dim=1)
    elif classification_mode == ClassificationMode.SINGLE_BINARY:
        probs = torch.sigmoid(outputs)
        return (probs > 0.5).float()
    elif classification_mode == ClassificationMode.MULTI_MULTICLASS:
        out_scs, out_lnfn, out_rnfn = outputs
        scs_pred  = torch.argmax(out_scs, dim=1, keepdim=True)
        lnfn_pred = torch.argmax(out_lnfn, dim=1, keepdim=True)
        rnfn_pred = torch.argmax(out_rnfn, dim=1, keepdim=True)
        return torch.cat([scs_pred, lnfn_pred, rnfn_pred], dim=1)
    else:  # MULTI_BINARY
        out_scs, out_lnfn, out_rnfn = outputs
        scs_probs  = torch.sigmoid(out_scs)
        lnfn_probs = torch.sigmoid(out_lnfn)
        rnfn_probs = torch.sigmoid(out_rnfn)
        scs_pred  = (scs_probs > 0.5).float()
        lnfn_pred = (lnfn_probs > 0.5).float()
        rnfn_pred = (rnfn_probs > 0.5).float()
        return torch.cat([scs_pred, lnfn_pred, rnfn_pred], dim=1)

# ---------------------------------------------------------------------
# 3) The cross_validate_model function (updated)
# ---------------------------------------------------------------------
def cross_validate_model(config_path="config.yml"):
    """
    Performs K-Fold cross-validation.
    For each fold:
      - Creates (train, val) splits.
      - Applies imbalance/WeightedRandomSampler if needed.
      - Trains for `num_epochs` with TQDM progress bars.
      - Saves per-epoch metrics to a CSV file for each fold.
      - Stores and evaluates the best model (based on validation balanced accuracy) on the validation set.
      - Logs additional metrics: macro-average precision, recall, f1-score, and per-disc reports including confusion matrices.
    
    Finally, aggregates and logs the average metrics across folds.
    """
    # --- 1) Load config + initial setup ---
    config = load_config(config_path)
    seed = config["project"]["seed"]
    use_k_fold = config["training"].get("use_k_fold", False)
    num_folds = config["training"].get("num_folds", 5)

    if not use_k_fold:
        print("[INFO] use_k_fold=false in config. Skipping cross-validation.")
        return

    # Create a dedicated CV results folder with naming convention.
    prefix = (
        f"cv_{config['training']['model_arch']}_"
        f"{config['training']['classification_mode']}_"
        f"{config['training']['selected_tensor_subfolder']}_"
        f"{config['training']['batch_size']}_"
        f"{config['training']['num_epochs']}_"
        f"{config['training']['learning_rate']}_"
        f"{config['training']['dropout_prob']}"
    )
    cv_dir = os.path.join("./models", prefix)
    os.makedirs(cv_dir, exist_ok=True)

    # Setup logger to save logs in cv_dir/training.log.
    logger = logging.getLogger(__name__)
    log_file_path = os.path.join(cv_dir, "training.log")
    setup_python_logger(logger, log_file_path)

    # Log dataset information from the config file.
    logger.info("===== DATASET INFORMATION FROM CONFIG =====")
    dataset_info = config.get("training", {})
    for key, value in dataset_info.items():
        logger.info(f"{key}: {value}")
    logger.info("============================================")

    logger.info("===== Starting K-Fold Cross-Validation =====")
    logger.info(f"Number of folds: {num_folds}, Seed={seed}")

    classification_str = config["training"]["classification_mode"]
    classification_mode = ClassificationMode(classification_str)
    binary_mapping_mode = config["training"].get("binary_mapping_mode", "normal_negative")
    disease = config["training"]["disease"]

    discs_for_training = config["training"].get("discs_for_training", ["L4/L5"])
    subfolder = config["training"]["selected_tensor_subfolder"]
    interim_base_path = config["data"]["interim_base_path"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    lr = config["training"]["learning_rate"]
    dropout_prob = config["training"]["dropout_prob"]
    early_stop = config["training"]["early_stopping_patience"]

    # Imbalance configuration.
    imbalance_cfg = config["training"].get("imbalance", {})
    method = imbalance_cfg.get("method", "none")
    undersampling_ratio = imbalance_cfg.get("undersampling_ratio", 0.5)
    oversampling_ratio = imbalance_cfg.get("oversampling_ratio", 1.0)
    smote = imbalance_cfg.get("smote", False)
    use_weighted_sampler = imbalance_cfg.get("use_weighted_sampler", False)

    # --- 2) Load the data ---
    merged_csv_path = os.path.join(config["data"]["raw_path"], "merged_train_data.csv")
    df_all = pd.read_csv(merged_csv_path)
    df_all["disc_label"] = df_all["level"].str.replace("/", "")

    # Filter for discs_for_training.
    df_filtered = df_all[df_all["level"].isin(discs_for_training)].copy()
    df_filtered = df_filtered.drop_duplicates(subset=["study_id", "level"]).copy()

    # Prepare labels for single-label vs multi-label.
    if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        label_col = f"{disease}_label"
        df_filtered = df_filtered.dropna(subset=[label_col])
        df_filtered[label_col] = df_filtered[label_col].astype(int)
        if classification_mode == ClassificationMode.SINGLE_BINARY:
            df_filtered[label_col] = df_filtered[label_col].apply(
                lambda x: map_label_3class_to_2class(x, mode=binary_mapping_mode)
            )
        final_label_key = label_col
    else:
        needed_cols = ["scs_label", "lnfn_label", "rnfn_label"]
        df_filtered = df_filtered.dropna(subset=needed_cols)
        for c in needed_cols:
            df_filtered[c] = df_filtered[c].astype(int)
        df_filtered["multi_label"] = (
            df_filtered["scs_label"].astype(str) + "_" +
            df_filtered["lnfn_label"].astype(str) + "_" +
            df_filtered["rnfn_label"].astype(str)
        )
        final_label_key = "multi_label"

    logger.info(f"[CrossVal] Filtered dataset shape = {df_filtered.shape}")

    # --- 3) Create K-Folds ---
    if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        y_strat = df_filtered[final_label_key].values
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(df_filtered, y_strat))
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(df_filtered))

    # To store metrics for each fold.
    fold_metrics = []

    # --- 4) Loop over folds ---
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        logger.info(f"\n=== [Fold {fold_idx}/{num_folds}] ===")

        # Sliced data.
        train_df = df_filtered.iloc[train_idx].copy()
        val_df   = df_filtered.iloc[val_idx].copy()

        # Apply imbalance strategy.
        if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
            balanced_train_df = apply_imbalance_strategy(
                df=train_df,
                label_col=final_label_key,
                method=method,
                undersampling_ratio=undersampling_ratio,
                oversampling_ratio=oversampling_ratio,
                smote=smote,
                is_multilabel=False
            )
        else:
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

        # Build datasets.
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
                parse_depth_from_folder(subfolder),
                parse_size_from_folder(subfolder),
                classification_mode=classification_str
            )
            val_ds = MultiLabelSpinalDatasetMultiDisc(
                val_df,
                build_disc_dirs_map(discs_for_training, subfolder, interim_base_path),
                parse_depth_from_folder(subfolder),
                parse_size_from_folder(subfolder),
                classification_mode=classification_str
            )

        # Build DataLoaders.
        if use_weighted_sampler:
            if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
                labels_list = train_df[final_label_key].tolist()
            else:
                labels_list = train_df["multi_label"].tolist()

            c = Counter(labels_list)
            total = len(labels_list)
            mapping = {lab: total / c[lab] for lab in c}
            sample_weights = [mapping[lab] for lab in labels_list]
            sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
            sampler = WeightedRandomSampler(sample_weights_tensor,
                                            num_samples=len(sample_weights_tensor),
                                            replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      sampler=sampler,
                                      collate_fn=custom_collate_filter_none)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      collate_fn=custom_collate_filter_none)
        # NOTE: It is assumed that the dataset returns a third element (disc ID) per sample.
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=custom_collate_filter_none)

        logger.info(f"[Fold {fold_idx}] Train size={len(train_ds)}, Val size={len(val_ds)}")

        # Build model & optimizer.
        model, criterion = build_model(config, in_channels)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        if config["training"].get("use_lr_scheduler", False):
            factor_ = config["training"].get("lr_scheduler_factor", 0.1)
            pat_ = config["training"].get("lr_scheduler_patience", 10)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor_, patience=pat_, verbose=True)

        best_fold_val_acc = 0.0
        best_fold_val_bal_acc = 0.0
        patience_counter = 0
        best_state = None  # To store the best model state for this fold

        # Lists to track per-epoch metrics.
        epoch_history = []
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        val_bal_acc_history = []

        # --- 4.6) Training loop with TQDM ---
        for epoch in range(num_epochs):
            # --- TRAIN ---
            model.train()
            train_loss_sum = 0.0
            train_correct = 0.0
            train_samples = 0

            for batch in tqdm(train_loader, desc=f"[Fold {fold_idx}][Train] Epoch {epoch+1}/{num_epochs}"):
                if batch is None:
                    continue
                x, y, _ = batch
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(x)
                preds = _to_predictions(outputs, classification_mode)
                if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
                    correct = (preds.cpu().numpy() == y.cpu().numpy()).sum()
                    train_correct += correct
                    train_samples += len(x)
                else:
                    arr_preds = preds.cpu().numpy()
                    arr_y = y.cpu().numpy()
                    for p_, y_ in zip(arr_preds, arr_y):
                        train_correct += np.sum(p_ == y_)
                    train_samples += len(x)

            avg_train_loss = train_loss_sum / (train_samples + 1e-8)
            if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
                train_acc = train_correct / (train_samples + 1e-8)
            else:
                train_acc = train_correct / (train_samples * 3 + 1e-8)

            # --- VALIDATION during training ---
            model.eval()
            val_loss_sum = 0.0
            val_samples = 0
            all_targets_train = []
            all_preds_train = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"[Fold {fold_idx}][Val] Epoch {epoch+1}/{num_epochs}"):
                    if batch is None:
                        continue
                    x_val, y_val, _ = batch
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    outputs_val = model(x_val)
                    val_loss = criterion(outputs_val, y_val)
                    val_loss_sum += val_loss.item() * len(x_val)
                    val_samples += len(x_val)

                    preds = _to_predictions(outputs_val, classification_mode)
                    all_targets_train.extend(y_val.cpu().numpy())
                    all_preds_train.extend(preds.cpu().numpy())

            avg_val_loss = val_loss_sum / (val_samples + 1e-8)
            if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
                correct_val = (np.array(all_preds_train) == np.array(all_targets_train)).sum()
                val_acc = correct_val / (val_samples + 1e-8)
            else:
                total_corr = 0
                total_elems = 0
                at_np = np.array(all_targets_train)
                ap_np = np.array(all_preds_train)
                for trow, prow in zip(at_np, ap_np):
                    total_corr += np.sum(trow == prow)
                    total_elems += len(trow)
                val_acc = total_corr / (total_elems + 1e-8)

            if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
                val_bal_acc = balanced_accuracy_score(all_targets_train, all_preds_train)
            else:
                at_np = np.array(all_targets_train)
                ap_np = np.array(all_preds_train)
                col_baccs = []
                for c_i in range(at_np.shape[1]):
                    col_baccs.append(balanced_accuracy_score(at_np[:, c_i], ap_np[:, c_i]))
                val_bal_acc = np.mean(col_baccs)

            if scheduler:
                scheduler.step(avg_val_loss)

            logger.info(
                f"[Fold {fold_idx}][Epoch {epoch+1}/{num_epochs}] "
                f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
                f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
                f"Val Bal Acc={val_bal_acc:.4f}"
            )

            # Record per-epoch metrics.
            epoch_history.append(epoch+1)
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            val_bal_acc_history.append(val_bal_acc)

            # Save best model state based on validation balanced accuracy.
            if val_bal_acc > best_fold_val_bal_acc:
                logger.info(f"[Fold {fold_idx}] New Best Balanced Accuracy: {val_bal_acc:.4f}")
                best_fold_val_bal_acc = val_bal_acc
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stop:
                    logger.info(f"[Fold {fold_idx}] Early stopping at epoch {epoch+1}")
                    break

        # --- 4.7) Evaluate the best model for this fold ---
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        all_targets = []
        all_preds = []
        all_discs = []  # to store disc identifiers per sample
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Fold {fold_idx}][Model Evaluation]"):
                if batch is None:
                    continue
                # Assuming the dataset returns (x, y, disc_id)
                x_val, y_val, disc = batch
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs_val = model(x_val)
                preds = _to_predictions(outputs_val, classification_mode)
                all_targets.extend(y_val.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_discs.extend(disc)  # store disc information

        # Compute overall classification report.
        cls_report_dict = classification_report(all_targets, all_preds, zero_division=0, digits=4, output_dict=True)
        macro_avg = cls_report_dict.get("macro avg", {})
        precision_macro = macro_avg.get("precision", 0.0)
        recall_macro = macro_avg.get("recall", 0.0)
        f1_macro = macro_avg.get("f1-score", 0.0)

        # Compute per-disc classification reports and confusion matrices.
        unique_discs = set(all_discs)
        per_disc_reports = {}
        per_disc_conf_matrices = {}
        for disc_id in unique_discs:
            indices = [i for i, d in enumerate(all_discs) if d == disc_id]
            disc_targets = [all_targets[i] for i in indices]
            disc_preds = [all_preds[i] for i in indices]
            report = classification_report(disc_targets, disc_preds, zero_division=0, digits=4)
            cm_disc = confusion_matrix(disc_targets, disc_preds)
            per_disc_reports[disc_id] = report
            per_disc_conf_matrices[disc_id] = cm_disc

        # Log overall metrics.
        if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
            cm_overall = confusion_matrix(all_targets, all_preds)
            logger.info(f"[Fold {fold_idx}] Best Model Overall Classification Report:\n{classification_report(all_targets, all_preds, zero_division=0, digits=4)}")
            logger.info(f"[Fold {fold_idx}] Overall Confusion Matrix:\n{cm_overall}")
        else:
            label_names = ["SCS", "LNFN", "RNFN"]
            at_np = np.array(all_targets)
            ap_np = np.array(all_preds)
            for i, name in enumerate(label_names):
                rep = classification_report(at_np[:, i], ap_np[:, i], zero_division=0, digits=4)
                cm_disc = confusion_matrix(at_np[:, i], ap_np[:, i])
                logger.info(f"[Fold {fold_idx}] {name} Overall Classification Report:\n{rep}")
                logger.info(f"[Fold {fold_idx}] {name} Overall Confusion Matrix:\n{cm_disc}")

        # Log per-disc reports and confusion matrices.
        for disc_id in unique_discs:
            logger.info(f"[Fold {fold_idx}] Classification Report for Disc {disc_id}:\n{per_disc_reports[disc_id]}")
            logger.info(f"[Fold {fold_idx}] Confusion Matrix for Disc {disc_id}:\n{per_disc_conf_matrices[disc_id]}")

        # Append fold metrics including the additional ones.
        fold_metrics.append({
            "fold_idx": fold_idx,
            "val_acc": val_acc,
            "val_bal_acc": val_bal_acc,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "per_disc_reports": per_disc_reports,
            "per_disc_conf_matrices": per_disc_conf_matrices
        })

        # Save per-epoch training/validation metrics to a CSV file for this fold.
        fold_df = pd.DataFrame({
            "epoch": epoch_history,
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "train_acc": train_acc_history,
            "val_acc": val_acc_history,
            "val_bal_acc": val_bal_acc_history
        })
        fold_csv_path = os.path.join(cv_dir, f"fold_{fold_idx}_metrics.csv")
        fold_df.to_csv(fold_csv_path, index=False)
        logger.info(f"[Fold {fold_idx}] Saved per-epoch metrics to {fold_csv_path}")

    # --- 5) Aggregate results across folds ---
    avg_acc = np.mean([fm["val_acc"] for fm in fold_metrics])
    avg_bal_acc = np.mean([fm["val_bal_acc"] for fm in fold_metrics])
    avg_precision = np.mean([fm["precision_macro"] for fm in fold_metrics])
    avg_recall = np.mean([fm["recall_macro"] for fm in fold_metrics])
    avg_f1 = np.mean([fm["f1_macro"] for fm in fold_metrics])

    logger.info("===== CROSS-VALIDATION RESULTS =====")
    for fm in fold_metrics:
        logger.info(
            f"Fold {fm['fold_idx']}: val_acc={fm['val_acc']:.4f}, val_bal_acc={fm['val_bal_acc']:.4f}, "
            f"Precision={fm['precision_macro']:.4f}, Recall={fm['recall_macro']:.4f}, F1={fm['f1_macro']:.4f}"
        )
    logger.info(
        f"[Average across {num_folds} folds] Accuracy={avg_acc:.4f}, Balanced Accuracy={avg_bal_acc:.4f}, "
        f"Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}"
    )
    logger.info("=====================================")

    print("[DONE] Cross-validation complete.")

if __name__ == "__main__":
    cross_validate_model("config.yml")
