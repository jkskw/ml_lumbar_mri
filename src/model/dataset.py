import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

def unify_3d_volumes_to_max_shape(volumes, pad_value=0.0):
    """
    Zero-pad each 3D volume (D,H,W) so that they all match the max D,H,W among them.
    """
    max_d = max(vol.shape[0] for vol in volumes)
    max_h = max(vol.shape[1] for vol in volumes)
    max_w = max(vol.shape[2] for vol in volumes)

    unified = []
    for vol in volumes:
        d, h, w = vol.shape
        pad_d = max_d - d
        pad_h = max_h - h
        pad_w = max_w - w
        # F.pad expects (left, right, top, bottom, front, back).
        vol_padded = F.pad(vol, (0, pad_w, 0, pad_h, 0, pad_d), value=pad_value)
        unified.append(vol_padded)
    return unified


class SingleDiseaseDatasetMultiDisc(Dataset):
    """
    Single-disease classification dataset that can handle multiple discs.
    Each row in `dataframe` must have a column `disc_label` that tells us 
    from which disc it comes (e.g., 'L4L5' or 'L5S1'), so we can pick the correct folder.

    Example subfolders (for 'scs'):
        ./data/interim/L4L5/target_window_128x128_5D_B2A2/scs/
        ./data/interim/L5S1/target_window_128x128_5D_B2A2/scs/
    """
    def __init__(
        self, 
        dataframe: pd.DataFrame,
        disc_dirs: dict,           # e.g. {"L4L5": "/data/interim/L4L5/...", "L5S1": "/data/interim/L5S1/..."}
        disease: str,              # "scs", "lnfn", etc.
        final_depth: int,
        final_size: tuple,
        classification_mode: str = "single_binary",
        transform=None,
        binary_mapping_mode: str = "normal_negative"  # "normal_negative" or "severe_positive"
    ):
        super().__init__()
        self.data = dataframe.reset_index(drop=True)
        self.disc_dirs = disc_dirs      # mapping disc_label -> subfolder path
        self.disease = disease
        self.classification_mode = classification_mode
        self.transform = transform
        self.binary_mapping_mode = binary_mapping_mode

        self.final_depth = final_depth
        self.final_size = final_size  # (height, width)
        self.expected_shape = (final_depth, final_size[0], final_size[1])

    def __len__(self):
        return len(self.data)

    def _to_binary(self, label: int) -> int:
        if self.binary_mapping_mode == "severe_positive":
            # Zakładamy, że etykiety 0, 1, 2 (normal, mild, moderate) będą traktowane jako negatywne,
            # a tylko pozostałe (np. 3, czyli severe) jako pozytywne.
            return 0 if label in [0, 1, 2] else 1
        else:
            # Domyślne: tylko etykieta 0 jest negatywna, a wszystkie inne pozytywne.
            return 0 if label == 0 else 1

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        study_id = int(row["study_id"])
        disc_label = row["disc_label"]

        disc_base_path = self.disc_dirs[disc_label]
        file_path = os.path.join(disc_base_path, self.disease, f"{study_id}.pt")

        if not os.path.exists(file_path):
            # Missing data => return None so collate can filter
            return None, None, None

        vol_3d = torch.load(file_path)  # shape: [D, H, W]
        if vol_3d.shape != self.expected_shape:
            return None, None, None

        # Unsqueeze channel => [1, D, H, W]
        vol_4d = vol_3d.unsqueeze(0)
        if self.transform:
            vol_4d = self.transform(vol_4d)

        # Build label
        raw_label = int(row[f"{self.disease}_label"])
        if self.classification_mode == "single_multiclass":
            label = torch.tensor(raw_label, dtype=torch.long)
        else:  # single_binary
            label = torch.tensor([self._to_binary(raw_label)], dtype=torch.float32)

        # Return disc_label as well, so we can do per-disc metrics in evaluation
        return (vol_4d, label, disc_label)


class MultiLabelSpinalDatasetMultiDisc(Dataset):
    """
    Multi-disease classification (multiclass or binary) that can handle multiple discs.
    For each disc_label in disc_dirs, we have subfolders:
        scs/, lnfn/, rnfn/
    """
    def __init__(
        self, 
        dataframe: pd.DataFrame,
        disc_dirs: dict,     # e.g. {"L4L5": "/data/interim/L4L5/..."}
        final_depth: int,
        final_size: tuple,
        classification_mode: str = "multi_multiclass",
        transform=None,
        binary_mapping_mode: str = "normal_negative"  # "normal_negative" or "severe_positive"
    ):
        super().__init__()
        self.data = dataframe.reset_index(drop=True)
        self.disc_dirs = disc_dirs
        self.classification_mode = classification_mode
        self.transform = transform
        self.binary_mapping_mode = binary_mapping_mode

        self.final_depth = final_depth
        self.final_size = final_size
        self.expected_shape = (final_depth, final_size[0], final_size[1])

    def __len__(self):
        return len(self.data)

    def _to_binary(self, label: int) -> int:
        if self.binary_mapping_mode == "severe_positive":
            return 0 if label in [0, 1] else 1
        else:
            return 0 if label == 0 else 1 

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        study_id = int(row["study_id"])
        disc_label = row["disc_label"]

        # For multi-disease, we load 3 separate .pt files:
        disc_base_path = self.disc_dirs[disc_label]
        scs_path  = os.path.join(disc_base_path, "scs",  f"{study_id}.pt")
        lnfn_path = os.path.join(disc_base_path, "lnfn", f"{study_id}.pt")
        rnfn_path = os.path.join(disc_base_path, "rnfn", f"{study_id}.pt")

        if (not os.path.exists(scs_path) or 
            not os.path.exists(lnfn_path) or 
            not os.path.exists(rnfn_path)):
            return None, None, None

        scs_t  = torch.load(scs_path)
        lnfn_t = torch.load(lnfn_path)
        rnfn_t = torch.load(rnfn_path)

        # Check shapes
        if (scs_t.shape != self.expected_shape or 
            lnfn_t.shape != self.expected_shape or 
            rnfn_t.shape != self.expected_shape):
            return None, None, None

        # Stack => [3, D, H, W]
        vol_4d = torch.stack([scs_t, lnfn_t, rnfn_t], dim=0)
        if self.transform:
            vol_4d = self.transform(vol_4d)

        # Build label [scs_label, lnfn_label, rnfn_label]
        scs_raw  = int(row["scs_label"])
        lnfn_raw = int(row["lnfn_label"])
        rnfn_raw = int(row["rnfn_label"])

        if self.classification_mode == "multi_multiclass":
            scs_lbl  = torch.tensor(scs_raw, dtype=torch.long)
            lnfn_lbl = torch.tensor(lnfn_raw, dtype=torch.long)
            rnfn_lbl = torch.tensor(rnfn_raw, dtype=torch.long)
        else:  # multi_binary
            scs_lbl  = torch.tensor(self._to_binary(scs_raw),  dtype=torch.float32)
            lnfn_lbl = torch.tensor(self._to_binary(lnfn_raw), dtype=torch.float32)
            rnfn_lbl = torch.tensor(self._to_binary(rnfn_raw), dtype=torch.float32)

        label_vec = torch.stack([scs_lbl, lnfn_lbl, rnfn_lbl])

        return (vol_4d, label_vec, disc_label)


def custom_collate_filter_none(batch):
    """
    Collate function that filters out None samples. 
    Some items might be (None, None, None) if shapes don't match or .pt files missing.
    """
    filtered = [item for item in batch if (item[0] is not None and item[1] is not None)]
    if len(filtered) == 0:
        return None
    xs, ys, discs = [], [], []
    for (x, y, dlabel) in filtered:
        xs.append(x)
        ys.append(y)
        discs.append(dlabel)

    out_x = torch.utils.data.default_collate(xs)
    out_y = torch.utils.data.default_collate(ys)
    return out_x, out_y, discs
