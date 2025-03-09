import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

def unify_3d_volumes_to_max_shape(volumes, pad_value=0.0):
    """
    Zero-pad each 3D volume (D,H,W) so that they all match the max D,H,W among them.
    
    volumes: List[torch.Tensor], each shape [D_i, H_i, W_i].
    pad_value: float value for padding.
    
    Returns: List[torch.Tensor] with unified shape [maxD, maxH, maxW].
    """
    # 1) Find max shape across all volumes
    max_d = 0
    max_h = 0
    max_w = 0
    for vol in volumes:
        d,h,w = vol.shape
        if d>max_d: max_d=d
        if h>max_h: max_h=h
        if w>max_w: max_w=w
    
    # 2) Pad each volume to [max_d, max_h, max_w]
    # F.pad expects (left, right, top, bottom, front, back) for 3D
    unified = []
    for vol in volumes:
        d,h,w = vol.shape
        pad_d = max_d - d
        pad_h = max_h - h
        pad_w = max_w - w
        # We'll only pad at the "end" of each dimension (front/bottom/right).
        # If you want symmetrical padding or something else, adjust accordingly.
        vol_padded = F.pad(vol, (0, pad_w, 0, pad_h, 0, pad_d), value=pad_value)
        unified.append(vol_padded)
    
    return unified

class SingleDiseaseDataset(Dataset):
    """
    Single-disease classification (multiclass or binary).
    Loads .pt files exactly as stored on disk, e.g. shape [D,H,W].
    Then unsqueezes channel => [1, D, H, W].
    """
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 disease: str,
                 tensor_dir: str,
                 final_depth: int,  
                 final_size: tuple,
                 classification_mode: str = "single_multiclass",
                 transform=None):
        super().__init__()
        self.data = dataframe.reset_index(drop=True)
        self.disease = disease
        self.tensor_dir = tensor_dir
        self.classification_mode = classification_mode
        self.transform = transform
        
        # Store expected dimensions from config.
        self.final_depth = final_depth
        self.final_size = final_size  # (height, width)
        # Expected tensor shape on disk: [D, H, W]
        self.expected_shape = (final_depth, final_size[0], final_size[1])

    def __len__(self):
        return len(self.data)

    def _load_tensor(self, study_id: int) -> torch.Tensor:
        file_path = os.path.join(self.tensor_dir, f"{int(study_id)}.pt")
        if os.path.exists(file_path):
            tensor = torch.load(file_path)  # shape: [D, H, W]
        else:
            tensor = torch.zeros((1, 1, 1), dtype=torch.float32)
        return tensor

    def _to_binary(self, label: int) -> int:
        return 0 if label == 0 else 1

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        study_id = row["study_id"]

        # 1) Load tensor from disk
        vol_3d = self._load_tensor(study_id)
        # Check the loaded tensor’s shape
        if vol_3d.shape != self.expected_shape:
            # print(f"[WARN] Skipping study_id={study_id}: tensor shape {vol_3d.shape} does not match expected {self.expected_shape}.")
            return None, None

        # 2) Unsqueeze channel dimension: [1, D, H, W]
        vol_4d = vol_3d.unsqueeze(0)
        if self.transform:
            vol_4d = self.transform(vol_4d)

        # 3) Build label
        raw_label = int(row[f"{self.disease}_label"])
        if self.classification_mode == "single_multiclass":
            label = torch.tensor(raw_label, dtype=torch.long)
        else:
            label = torch.tensor([self._to_binary(raw_label)], dtype=torch.float32)

        return vol_4d, label


class MultiLabelSpinalDataset(Dataset):
    """
    Multi-disease classification (multiclass or binary).
    Loads 3 separate .pt files (scs, lnfn, rnfn) from disk, each e.g. [D,H,W].
    Stacks => [3, D, H, W].
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 scs_dir: str,
                 lnfn_dir: str,
                 rnfn_dir: str,
                 final_depth: int,
                 final_size: tuple,
                 classification_mode: str = "multi_multiclass",
                 transform=None):
        super().__init__()
        self.data = dataframe.reset_index(drop=True)
        self.scs_dir = scs_dir
        self.lnfn_dir = lnfn_dir
        self.rnfn_dir = rnfn_dir
        self.classification_mode = classification_mode
        self.transform = transform

        self.final_depth = final_depth
        self.final_size = final_size  # (height, width)
        self.expected_shape = (final_depth, final_size[0], final_size[1])

    def __len__(self):
        return len(self.data)

    def _load_tensor(self, directory: str, study_id: int) -> torch.Tensor:
        file_path = os.path.join(directory, f"{int(study_id)}.pt")
        if os.path.exists(file_path):
            tensor = torch.load(file_path)  # shape: [D, H, W]
        else:
            tensor = torch.zeros((1, 1, 1), dtype=torch.float32)
        return tensor

    def _to_binary(self, label: int) -> int:
        return 0 if label == 0 else 1

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        study_id = row["study_id"]

        # Load each tensor
        scs_t  = self._load_tensor(self.scs_dir,  study_id)
        lnfn_t = self._load_tensor(self.lnfn_dir, study_id)
        rnfn_t = self._load_tensor(self.rnfn_dir, study_id)

        # Check each tensor’s shape
        if scs_t.shape != self.expected_shape:
            # print(f"[WARN] Skipping study_id={study_id} (scs): tensor shape {scs_t.shape} does not match expected {self.expected_shape}.")
            return None, None
        if lnfn_t.shape != self.expected_shape:
            # print(f"[WARN] Skipping study_id={study_id} (lnfn): tensor shape {lnfn_t.shape} does not match expected {self.expected_shape}.")
            return None, None
        if rnfn_t.shape != self.expected_shape:
            # print(f"[WARN] Skipping study_id={study_id} (rnfn): tensor shape {rnfn_t.shape} does not match expected {self.expected_shape}.")
            return None, None

        # Optionally, you can still unify shapes if needed
        scs_t, lnfn_t, rnfn_t = unify_3d_volumes_to_max_shape([scs_t, lnfn_t, rnfn_t])
    
        # Stack into a 4D tensor: [3, D, H, W]
        vol_4d = torch.stack([scs_t, lnfn_t, rnfn_t], dim=0)
        if self.transform:
            vol_4d = self.transform(vol_4d)

        # Build label vector [scs_label, lnfn_label, rnfn_label]
        scs_raw  = int(row["scs_label"])
        lnfn_raw = int(row["lnfn_label"])
        rnfn_raw = int(row["rnfn_label"])
        if self.classification_mode == "multi_multiclass":
            scs_lbl  = torch.tensor(scs_raw,  dtype=torch.long)
            lnfn_lbl = torch.tensor(lnfn_raw, dtype=torch.long)
            rnfn_lbl = torch.tensor(rnfn_raw, dtype=torch.long)
        else:
            scs_lbl  = torch.tensor(self._to_binary(scs_raw),  dtype=torch.float32)
            lnfn_lbl = torch.tensor(self._to_binary(lnfn_raw), dtype=torch.float32)
            rnfn_lbl = torch.tensor(self._to_binary(rnfn_raw), dtype=torch.float32)
        label_1d = torch.stack([scs_lbl, lnfn_lbl, rnfn_lbl])  # shape: [3]

        # Also check the stacked tensor’s overall shape
        expected_stacked_shape = (3, self.expected_shape[0], self.expected_shape[1], self.expected_shape[2])
        if vol_4d.shape != expected_stacked_shape:
            # print(f"[WARN] Skipping study_id={study_id}: stacked tensor shape {vol_4d.shape} does not match expected {expected_stacked_shape}.")
            return None, None

        return vol_4d, label_1d

def custom_collate_filter_none(batch):
    # Filter out samples that are None.
    filtered = [item for item in batch if item[0] is not None and item[1] is not None]
    if len(filtered) == 0:
        # If every sample in the batch was discarded, return an empty batch.
        return None
    return torch.utils.data.default_collate(filtered)