import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pydicom as dcm
from glob import glob
from tqdm.auto import tqdm

class DataPreprocessor:
    """
    A class to preprocess lumbar MRI DICOM images for multiple discs, building
    scs_label, lnfn_label, rnfn_label columns dynamically.

    It:
      1) Merges train_series_descriptions.csv + train_label_coordinates.csv => preliminary df
      2) Merges classification labels from train.csv for *all discs*:
         e.g. spinal_canal_stenosis_l1_l2, l2_l3, l4_l5, l5_s1, etc.
      3) Unifies them into single scs_label, lnfn_label, rnfn_label columns
         based on each row's "level".
      4) Filters out Axial T2, focuses on the discs in `discs_to_process`.
      5) Saves .pt volumes in subfolders scs/, lnfn/, rnfn/.
    """

    def __init__(self, config_path: str = "config.yml"):
        self.config = self.load_config(config_path)
        self.raw_path = self.config["data"]["raw_path"]
        self.interim_base_path = self.config["data"]["interim_base_path"]

        prep_cfg = self.config["preprocessing"]
        self.discs_to_process = prep_cfg["discs_to_process"]
        self.final_depth = prep_cfg["final_depth"]
        self.final_size = tuple(prep_cfg["final_size"])
        self.cropped_size = tuple(prep_cfg["cropped_size"])
        self.cropping_dict = prep_cfg["cropping"]
        self.mode = prep_cfg["mode"]

        tw = prep_cfg["target_window"]
        self.slices_before = tw["slices_before"]
        self.slices_after = tw["slices_after"]

        # Decide subfolder name, e.g. target_window_128x128_5D_B2A2
        if self.mode == "full_series":
            self.output_subfolder = f"full_series_{self.final_size[0]}x{self.final_size[1]}_{self.final_depth}D"
        else:
            depth = self.slices_before + self.slices_after + 1
            self.output_subfolder = f"target_window_{self.final_size[0]}x{self.final_size[1]}_{depth}D_B{self.slices_before}A{self.slices_after}"

        print(f"[INFO] DataPreprocessor init: building volumes in subfolders like {self.output_subfolder}")

    def load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def disc_to_suffix(self, level_str: str) -> str:
        """
        Converts a level like 'L4/L5' -> 'l4_l5', 'L5/S1' -> 'l5_s1'.
        We'll use this to find the right column in train.csv 
        e.g. 'spinal_canal_stenosis_l5_s1'.
        """
        return level_str.lower().replace("/", "_")

    def build_merged_train_data(self) -> pd.DataFrame:
        """
        Builds or loads 'merged_train_data.csv'.
        If the file already exists, loads it directly.

        Otherwise:
          1) merges train_series_descriptions.csv + train_label_coordinates.csv 
             => merged_df
          2) merges *all* columns from train.csv 
             => e.g. spinal_canal_stenosis_l5_s1, etc.
          3) for each row, unify them into scs_label, lnfn_label, rnfn_label 
             using row['level'].
          4) writes merged_train_data.csv.
        """
        merged_csv_path = os.path.join(self.raw_path, "merged_train_data.csv")
        if os.path.exists(merged_csv_path):
            print(f"[INFO] Using existing {merged_csv_path}")
            return pd.read_csv(merged_csv_path)

        # read series + label coords
        series_csv = os.path.join(self.raw_path, "train_series_descriptions.csv")
        labels_csv = os.path.join(self.raw_path, "train_label_coordinates.csv")

        df_series = pd.read_csv(series_csv)
        df_labels = pd.read_csv(labels_csv)
        merged_df = pd.merge(df_labels, df_series, on=["study_id","series_id"], how="inner")

        # compute shape from DICOM
        shapes = []
        for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Computing DICOM shapes"):
            h, w = self.get_dicom_shape(row["study_id"], row["series_id"], row["instance_number"])
            shapes.append((h,w))
        merged_df["height"] = [s[0] for s in shapes]
        merged_df["width"]  = [s[1] for s in shapes]
        merged_df.dropna(subset=["height","width"], inplace=True)
        merged_df["height"] = merged_df["height"].astype(int)
        merged_df["width"]  = merged_df["width"].astype(int)

        # read train.csv with columns for *all discs*
        train_csv = os.path.join(self.raw_path, "train.csv")
        df_train = pd.read_csv(train_csv)  # must contain e.g. spinal_canal_stenosis_l5_s1, etc.

        # Merge them => now merged_df has all columns from train.csv
        merged_df = pd.merge(merged_df, df_train, on="study_id", how="left")

        # define the text->int label map for the degenerative labels
        label_map = {
            "Normal/Mild": 0,
            "Moderate": 1,
            "Severe": 2
        }

        # define pickers
        def pick_scs_label(row):
            disc_suffix = self.disc_to_suffix(row["level"])   # e.g. "l5_s1"
            col_name = f"spinal_canal_stenosis_{disc_suffix}" # e.g. "spinal_canal_stenosis_l5_s1"
            if col_name not in row:
                return None
            raw_val = row[col_name]  # might be "Moderate", or NaN
            return label_map.get(raw_val, None)

        def pick_lnfn_label(row):
            disc_suffix = self.disc_to_suffix(row["level"])
            col_name = f"left_neural_foraminal_narrowing_{disc_suffix}"
            if col_name not in row:
                return None
            raw_val = row[col_name]
            return label_map.get(raw_val, None)

        def pick_rnfn_label(row):
            disc_suffix = self.disc_to_suffix(row["level"])
            col_name = f"right_neural_foraminal_narrowing_{disc_suffix}"
            if col_name not in row:
                return None
            raw_val = row[col_name]
            return label_map.get(raw_val, None)

        merged_df["scs_label"]  = merged_df.apply(pick_scs_label, axis=1)
        merged_df["lnfn_label"] = merged_df.apply(pick_lnfn_label, axis=1)
        merged_df["rnfn_label"] = merged_df.apply(pick_rnfn_label, axis=1)

        merged_df.to_csv(merged_csv_path, index=False)
        print(f"[INFO] Wrote merged_train_data.csv with shape={merged_df.shape}")
        return merged_df

    def get_dicom_shape(self, study_id, series_id, instance_num):
        # read actual DICOM to get shape
        dcm_path = os.path.join(self.raw_path, "train_images", str(study_id), str(series_id), f"{int(instance_num)}.dcm")
        if not os.path.isfile(dcm_path):
            return None, None
        ds = dcm.dcmread(dcm_path)
        arr = ds.pixel_array
        return arr.shape[0], arr.shape[1]

    def load_and_normalize_dicom(self, dicom_path):
        ds = dcm.dcmread(dicom_path)
        arr = ds.pixel_array.astype(np.float32)
        t = torch.from_numpy(arr)
        p01 = torch.quantile(t, 0.01)
        p99 = torch.quantile(t, 0.99)
        t = torch.clamp(t, p01, p99)
        t -= t.min()
        t /= (t.max() + 1e-6)
        return t

    def resize_2d(self, image: torch.Tensor, out_h: int, out_w: int):
        # simple 2D bilinear
        image = image.unsqueeze(0).unsqueeze(0)
        out   = F.interpolate(image, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return out.squeeze(0).squeeze(0)

    def scale_xy(self, x, y, orig_w, orig_h, new_w, new_h):
        return x * (new_w / orig_w), y * (new_h / orig_h)

    def crop_around_xy(self, image, cx, cy, left, right, top, bottom):
        H, W = image.shape
        row_min = max(int(cy - top), 0)
        row_max = min(int(cy + bottom), H)
        col_min = max(int(cx - left), 0)
        col_max = min(int(cx + right), W)
        if row_min>=row_max or col_min>=col_max:
            return torch.zeros((0,0), dtype=image.dtype)
        return image[row_min:row_max, col_min:col_max]

    def pad_or_resize_3d(self, vol_3d, out_depth, out_hw):
        D,H,W = vol_3d.shape
        if D < out_depth:
            extra = out_depth - D
            pad_vol = torch.zeros((extra, H, W), dtype=vol_3d.dtype)
            vol_3d = torch.cat([vol_3d, pad_vol], dim=0)
        elif D > out_depth:
            vol_3d = F.interpolate(vol_3d.unsqueeze(0).unsqueeze(0), size=(out_depth,H,W),
                                   mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
        final_h, final_w = out_hw
        vol_3d = F.interpolate(vol_3d.unsqueeze(0).unsqueeze(0), size=(vol_3d.shape[0], final_h, final_w),
                               mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
        return vol_3d

    def get_crop_margins(self, series_desc: str):
        # example logic from your code
        if series_desc == "Sagittal T2/STIR":
            m = self.cropping_dict.get("scs", {}).get("sagt2", {})
        elif series_desc == "Sagittal T1":
            m = self.cropping_dict.get("nfn", {}).get("sagt1", {})
        else:
            m = {}
        return (m.get("left", 0), m.get("right", 0), m.get("upper",0), m.get("lower",0))

    def get_subfolder_name(self, condition: str):
        if condition == "Spinal Canal Stenosis":
            return "scs"
        elif condition == "Right Neural Foraminal Narrowing":
            return "rnfn"
        elif condition == "Left Neural Foraminal Narrowing":
            return "lnfn"
        return "other"

    def process_all_discs(self):
        """
        Main entry point:
          - build or load merged_train_data (with dynamic scs_label, lnfn_label, rnfn_label)
          - for each disc in discs_to_process, filter, build volumes in subfolders
        """
        merged_df = self.build_merged_train_data()

        for disc_label in self.discs_to_process:
            self.process_single_disc(merged_df, disc_label)

    def process_single_disc(self, merged_df: pd.DataFrame, disc_label: str):
        """
        Preprocess a single disc (e.g. "L4/L5" or "L5/S1") => build .pt volumes
        in e.g. ./data/interim/L5S1/<output_subfolder>/scs/<study_id>.pt, etc.
        """

        df_filt = merged_df.copy()
        df_filt = df_filt[df_filt["level"] == disc_label]
        df_filt = df_filt[df_filt["series_description"] != "Axial T2"]

        # Build disc output dir: e.g. ./data/interim/L5S1/target_window_128x128_5D_B2A2
        out_dir = os.path.join(self.interim_base_path,
                               disc_label.replace("/", ""),  # "L5S1"
                               self.output_subfolder)
        os.makedirs(out_dir, exist_ok=True)

        group_cols = ["study_id", "series_id", "condition"]
        grouped = df_filt.groupby(group_cols)

        for (sid, serid, cond), group in tqdm(grouped, desc=f"Building volumes for {disc_label}"):
            row0 = group.iloc[0]
            series_desc = row0["series_description"]
            x_val, y_val = row0["x"], row0["y"]
            orig_w, orig_h = row0["width"], row0["height"]
            target_inst = row0["instance_number"]

            left, right, up, down = self.get_crop_margins(series_desc)
            if (left+right+up+down) == 0:
                continue

            dcm_folder = os.path.join(self.raw_path, "train_images", str(sid), str(serid))
            dcm_paths = sorted(glob(os.path.join(dcm_folder, "*.dcm")),
                               key=lambda p: int(os.path.basename(p).replace(".dcm","")))

            # find target index
            target_idx = None
            for i,dp in enumerate(dcm_paths):
                if int(os.path.basename(dp).replace(".dcm","")) == target_inst:
                    target_idx = i
                    break
            if target_idx is None:
                continue

            # pick slices
            if self.mode.lower() == "target_window":
                start_idx = max(0, target_idx - self.slices_before)
                end_idx   = min(len(dcm_paths)-1, target_idx + self.slices_after)
                selected_paths = dcm_paths[start_idx:end_idx+1]
            else:
                selected_paths = dcm_paths  # or do full_series logic

            # load slices
            slices2d = []
            for dp in selected_paths:
                img = self.load_and_normalize_dicom(dp)
                img = self.resize_2d(img, self.cropped_size[0], self.cropped_size[1])
                # scale x,y
                x_scl, y_scl = self.scale_xy(x_val, y_val, orig_w, orig_h,
                                             self.cropped_size[1], self.cropped_size[0])
                patch = self.crop_around_xy(img, x_scl, y_scl, left,right,up,down)
                if patch.shape[0]*patch.shape[1] == 0:
                    continue
                slices2d.append(patch)

            if not slices2d:
                continue

            # unify
            max_h = max(s.shape[0] for s in slices2d)
            max_w = max(s.shape[1] for s in slices2d)
            padded_stack = []
            for s in slices2d:
                ph = max_h - s.shape[0]
                pw = max_w - s.shape[1]
                s_padded = F.pad(s, (0, pw, 0, ph), value=0.0)
                padded_stack.append(s_padded)
            vol_3d = torch.stack(padded_stack, dim=0)  # [D, H, W]

            # if full_series => pad or resize to self.final_depth
            if self.mode.lower() == "full_series":
                vol_3d = self.pad_or_resize_3d(vol_3d, self.final_depth, self.final_size)
            else:
                # target_window => keep depth but fix H,W
                vol_3d = F.interpolate(vol_3d.unsqueeze(0).unsqueeze(0),
                                       size=(vol_3d.shape[0], self.final_size[0], self.final_size[1]),
                                       mode='trilinear', align_corners=False
                                      ).squeeze(0).squeeze(0)

            # write
            subfolder = self.get_subfolder_name(cond)  # e.g. scs, lnfn, rnfn
            cond_dir = os.path.join(out_dir, subfolder)
            os.makedirs(cond_dir, exist_ok=True)
            out_path = os.path.join(cond_dir, f"{sid}.pt")
            torch.save(vol_3d, out_path)

    def process(self):
        """
        If you just want to run them all in one function:
        """
        self.process_all_discs()

if __name__ == "__main__":
    preprocessor = DataPreprocessor("config.yml")
    preprocessor.process()
