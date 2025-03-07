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
                 # The next two are no longer forced in code, but we keep them for your constructor signature
                 final_depth: int,  
                 final_size: tuple,
                 classification_mode: str = "single_multiclass",
                 transform=None):
        """
        Args:
            dataframe: Must have "study_id" and a column like "<disease>_label".
            disease: e.g. "lnfn", "scs", or "rnfn".
            tensor_dir: Path to .pt files for that disease.
            final_depth, final_size: Not actually used to override shapes, but retained for signature.
            classification_mode: "single_multiclass" or "single_binary".
            transform: Optional transform on the 4D volume.
        """
        super().__init__()
        self.data = dataframe.reset_index(drop=True)
        self.disease = disease
        self.tensor_dir = tensor_dir
        self.classification_mode = classification_mode
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _load_tensor(self, study_id: int) -> torch.Tensor:
        file_path = os.path.join(self.tensor_dir, f"{int(study_id)}.pt")
        if os.path.exists(file_path):
            # Physically stored shape, e.g. [D, H, W]
            tensor = torch.load(file_path)
        else:
            # If missing, fallback to an empty volume or minimal shape.
            # e.g. shape [1,1], though you might want [0,0] or some other fallback.
            tensor = torch.zeros((1,1,1), dtype=torch.float32)
        return tensor

    def _to_binary(self, label: int) -> int:
        # For single_binary: 0 => 0, else => 1
        return 0 if label == 0 else 1

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        study_id = row["study_id"]

        # 1) Load the disk shape [D,H,W].
        vol_3d = self._load_tensor(study_id)

        # 2) Insert channel dimension => [1, D, H, W]
        vol_4d = vol_3d.unsqueeze(0)

        if self.transform:
            vol_4d = self.transform(vol_4d)

        # 3) Build label
        raw_label = int(row[f"{self.disease}_label"])
        if self.classification_mode == "single_multiclass":
            label = torch.tensor(raw_label, dtype=torch.long)  # e.g. 0..2
        else:
            # single_binary
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
        """
        Args:
            dataframe: Must have "study_id", "scs_label", "lnfn_label", "rnfn_label".
            scs_dir, lnfn_dir, rnfn_dir: Paths to .pt files for each condition.
            final_depth, final_size: Not forcibly used to override shapes, but retained for signature.
            classification_mode: "multi_multiclass" or "multi_binary".
            transform: optional transform on the 4D volume.
        """
        super().__init__()
        self.data = dataframe.reset_index(drop=True)
        self.scs_dir = scs_dir
        self.lnfn_dir = lnfn_dir
        self.rnfn_dir = rnfn_dir
        self.classification_mode = classification_mode
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _load_tensor(self, directory: str, study_id: int) -> torch.Tensor:
        file_path = os.path.join(directory, f"{int(study_id)}.pt")
        if os.path.exists(file_path):
            # e.g. shape [D,H,W]
            tensor = torch.load(file_path)
        else:
            # Fallback minimal shape
            tensor = torch.zeros((1,1,1), dtype=torch.float32)
        return tensor

    def _to_binary(self, label: int) -> int:
        return 0 if label == 0 else 1

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        study_id = row["study_id"]

        # 1) Load each .pt => e.g. [D,H,W]
        scs_t  = self._load_tensor(self.scs_dir,  study_id)
        lnfn_t = self._load_tensor(self.lnfn_dir, study_id)
        rnfn_t = self._load_tensor(self.rnfn_dir, study_id)

        # if not( scs_t.shape == lnfn_t.shape == rnfn_t.shape ):
        # # You can either 
        # # 1) raise an exception that the DataLoader will catch
        # # 2) or return None, None and handle in __len__, etc.
        #     return None, None

        scs_t, lnfn_t, rnfn_t = unify_3d_volumes_to_max_shape([scs_t, lnfn_t, rnfn_t])

    
        # 2) Stack => shape [3, D, H, W]
        vol_4d = torch.stack([scs_t, lnfn_t, rnfn_t], dim=0)

        if self.transform:
            vol_4d = self.transform(vol_4d)

        # 3) Build label => shape [3]
        scs_raw  = int(row["scs_label"])
        lnfn_raw = int(row["lnfn_label"])
        rnfn_raw = int(row["rnfn_label"])

        if self.classification_mode == "multi_multiclass":
            scs_lbl  = torch.tensor(scs_raw,  dtype=torch.long)
            lnfn_lbl = torch.tensor(lnfn_raw, dtype=torch.long)
            rnfn_lbl = torch.tensor(rnfn_raw, dtype=torch.long)
        else:
            # multi_binary => 0 vs 1
            scs_lbl  = torch.tensor(self._to_binary(scs_raw),  dtype=torch.float32)
            lnfn_lbl = torch.tensor(self._to_binary(lnfn_raw), dtype=torch.float32)
            rnfn_lbl = torch.tensor(self._to_binary(rnfn_raw), dtype=torch.float32)

        label_1d = torch.stack([scs_lbl, lnfn_lbl, rnfn_lbl])  # shape [3]

        return vol_4d, label_1d
