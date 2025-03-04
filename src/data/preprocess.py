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
    A class to preprocess lumbar MRI DICOM images by:
      1) Merging train_series_descriptions.csv and train_label_coordinates.csv
         into a single merged DataFrame (with height/width from actual DICOM files).
      2) Merging classification labels (Normal/Mild->0, Moderate->1, Severe->2) from train.csv
         for L4/L5 disc:
           - spinal_canal_stenosis_l4_l5
           - left_neural_foraminal_narrowing_l4_l5
           - right_neural_foraminal_narrowing_l4_l5
      3) Saving that final merged DataFrame to merged_train_data.csv (containing all needed columns)
         ONLY if it doesn't exist already; otherwise we load it directly.
      4) Filtering out Axial T2, focusing on the target disc, building .pt volumes in
         scs / lnfn / rnfn subfolders (depending on 'condition').
    """

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.raw_path = self.config["data"].get("raw_path", "./data/raw")
        self.tensors_base_path = self.config["data"].get(
            "interim_path", "./data/interim")

        # Preprocessing settings
        prep_cfg = self.config["preprocessing"]
        self.target_disc = prep_cfg.get("target_disc", "L4/L5")
        self.final_depth = prep_cfg.get("final_depth", 32)
        self.final_size = tuple(prep_cfg.get("final_size", [96, 96]))
        self.cropped_size = tuple(prep_cfg.get("cropped_size", [512, 512]))
        self.cropping_dict = prep_cfg.get("cropping", {})

        # Mode: "full_series" or "target_window"
        self.preprocessing_mode = prep_cfg.get(
            "mode", "full_series")
        tw_cfg = self.config.get("target_window", {})
        self.slices_before = tw_cfg.get("slices_before", 0)
        self.slices_after = tw_cfg.get("slices_after", 0)

        # Decide output subdirectory name
        if self.preprocessing_mode == "full_series":
            output_subdir = f"{self.preprocessing_mode}_{self.final_size[0]}x{self.final_size[1]}_{self.final_depth}D"
        else:
            output_subdir = f"{self.preprocessing_mode}_{self.final_size[0]}x{self.final_size[1]}_{int(self.slices_before) + int(self.slices_after) + 1}D_B{self.slices_before}A{self.slices_after}"

        self.output_dir = os.path.join(self.tensors_base_path, output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[INFO] Processed tensors will be saved in: {self.output_dir}")

    def load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print("[INFO] Loaded configuration:")
        print(config)
        return config

    def build_merged_train_data(self) -> pd.DataFrame:
        """
        Builds or loads 'merged_train_data.csv'.
        If the file already exists, we load and return it.
        Otherwise:
          1) Merges train_series_descriptions.csv + train_label_coordinates.csv => preliminary df
          2) Adds [height, width] from DICOM scanning
          3) Merges classification columns for L4/L5 from train.csv => numeric columns scs_label, lnfn_label, rnfn_label
          4) Saves the final DataFrame to 'merged_train_data.csv' in data/raw
          5) Returns that DataFrame
        """
        merged_csv_path = os.path.join(self.raw_path, "merged_train_data.csv")
        if os.path.isfile(merged_csv_path):
            print(
                f"[INFO] '{merged_csv_path}' already exists. Loading it directly.")
            return pd.read_csv(merged_csv_path)

        # (a) read series and label coordinates
        series_csv = os.path.join(
            self.raw_path, "train_series_descriptions.csv")
        labels_csv = os.path.join(self.raw_path, "train_label_coordinates.csv")

        df_series = pd.read_csv(series_csv)
        df_labels = pd.read_csv(labels_csv)
        merged_df = pd.merge(df_labels, df_series, on=[
                             "study_id", "series_id"], how="inner")
        print(f"[INFO] Merged shape before DICOM shape: {merged_df.shape}")

        # (b) compute shape
        shapes = []
        for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Computing DICOM shapes"):
            h, w = self.get_dicom_shape(
                row["study_id"], row["series_id"], row["instance_number"])
            shapes.append((h, w))
        merged_df["height"] = [s[0] for s in shapes]
        merged_df["width"] = [s[1] for s in shapes]
        merged_df.dropna(subset=["height", "width"], inplace=True)
        merged_df["height"] = merged_df["height"].astype(int)
        merged_df["width"] = merged_df["width"].astype(int)

        # (c) merge classification labels from train.csv for L4/L5
        train_csv = os.path.join(self.raw_path, "train.csv")
        if os.path.isfile(train_csv):
            df_train = pd.read_csv(train_csv)
            keep_cols = [
                "study_id",
                "spinal_canal_stenosis_l4_l5",
                "left_neural_foraminal_narrowing_l4_l5",
                "right_neural_foraminal_narrowing_l4_l5",
            ]
            df_train = df_train[keep_cols].copy()

            # map text -> numeric
            label_map = {
                "Normal/Mild": 0,
                "Moderate": 1,
                "Severe": 2
            }
            df_train["scs_label"] = df_train["spinal_canal_stenosis_l4_l5"].map(
                label_map)
            df_train["lnfn_label"] = df_train["left_neural_foraminal_narrowing_l4_l5"].map(
                label_map)
            df_train["rnfn_label"] = df_train["right_neural_foraminal_narrowing_l4_l5"].map(
                label_map)

            df_train.drop(
                columns=["spinal_canal_stenosis_l4_l5", "left_neural_foraminal_narrowing_l4_l5",
                         "right_neural_foraminal_narrowing_l4_l5"],
                inplace=True
            )

            merged_df = pd.merge(merged_df, df_train,
                                 on="study_id", how="left")
        else:
            print(
                f"[WARNING] {train_csv} not found, skipping classification label merge.")

        merged_df.to_csv(merged_csv_path, index=False)
        print(
            f"[INFO] Final merged CSV with classification saved to {merged_csv_path}, shape={merged_df.shape}")
        return merged_df

    def get_dicom_shape(self, study_id: int, series_id: int, instance_number: int):
        dcm_path = os.path.join(
            self.raw_path, "train_images", str(study_id), str(series_id),
            f"{int(instance_number)}.dcm"
        )
        if not os.path.isfile(dcm_path):
            return None, None
        ds = dcm.dcmread(dcm_path)
        arr = ds.pixel_array
        return arr.shape[0], arr.shape[1]

    def load_and_normalize_dicom(self, dicom_path: str) -> torch.Tensor:
        ds = dcm.dcmread(dicom_path)
        arr = ds.pixel_array.astype(np.float32)
        t = torch.from_numpy(arr)
        p01 = torch.quantile(t, 0.01)
        p99 = torch.quantile(t, 0.99)
        t = torch.clamp(t, p01, p99)
        t -= t.min()
        t /= (t.max() + 1e-6)
        return t

    def resize_2d(self, image: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        image = image.unsqueeze(0).unsqueeze(0)  # [B=1, C=1, H, W]
        resized = F.interpolate(image, size=(
            out_h, out_w), mode='bilinear', align_corners=False)
        return resized.squeeze(0).squeeze(0)

    def crop_around_xy(self, image: torch.Tensor, cx: float, cy: float, left: int, right: int, top: int, bottom: int) -> torch.Tensor:
        H, W = image.shape
        row_min = max(int(cy - top), 0)
        row_max = min(int(cy + bottom), H)
        col_min = max(int(cx - left), 0)
        col_max = min(int(cx + right), W)
        if row_min >= row_max or col_min >= col_max:
            return torch.zeros((0, 0), dtype=image.dtype)
        return image[row_min:row_max, col_min:col_max]

    def pad_or_resize_3d(self, volume: torch.Tensor, out_depth: int, final_hw: tuple) -> torch.Tensor:
        D, H, W = volume.shape
        # Depth
        if D < out_depth:
            pad_depth = out_depth - D
            volume = torch.cat([volume, torch.zeros(
                (pad_depth, H, W), dtype=volume.dtype)], dim=0)
        elif D > out_depth:
            volume = F.interpolate(
                volume.unsqueeze(0).unsqueeze(0),
                size=(out_depth, H, W),
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        # Then resize H,W
        final_h, final_w = final_hw
        volume = F.interpolate(
            volume.unsqueeze(0).unsqueeze(0),
            size=(volume.shape[0], final_h, final_w),
            mode='trilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        return volume

    def scale_xy(self, x: float, y: float, orig_w: int, orig_h: int, new_w: int, new_h: int):
        return x * (new_w / orig_w), y * (new_h / orig_h)

    def get_crop_margins(self, series_desc: str):
        if series_desc == "Sagittal T2/STIR":
            margins = self.cropping_dict.get("scs", {}).get("sagt2", {})
        elif series_desc == "Sagittal T1":
            margins = self.cropping_dict.get("nfn", {}).get("sagt1", {})
        else:
            margins = {}
        return (
            margins.get("left", 0),
            margins.get("right", 0),
            margins.get("upper", 0),
            margins.get("lower", 0),
        )

    def get_subfolder_name(self, condition: str) -> str:
        if condition == "Spinal Canal Stenosis":
            return "scs"
        elif condition == "Right Neural Foraminal Narrowing":
            return "rnfn"
        elif condition == "Left Neural Foraminal Narrowing":
            return "lnfn"
        return "other"

    def process(self):
        """
        Steps:
          (1) Build (or load) merged_train_data.csv with classification columns.
          (2) Read it, do final filtering (Axial T2, target disc).
          (3) For each group, build 3D volume and store in subfolder.
          (4) Overwrite final CSV with the same data, so user sees final shape.
        """
        # Step 1: build or load merged_train_data
        merged_csv_path = os.path.join(self.raw_path, "merged_train_data.csv")
        merged_df = self.build_merged_train_data()

        # Step 2: filter out Axial T2, keep only target disc
        df_filtered = merged_df.copy()
        df_filtered = df_filtered[df_filtered["series_description"] != "Axial T2"]
        df_filtered = df_filtered[df_filtered["level"] == self.target_disc]
        print(
            f"[INFO] After disc & Axial filtering: shape={df_filtered.shape}")

        # Group by (study_id, series_id, condition)
        group_cols = ["study_id", "series_id", "condition"]
        grouped = df_filtered.groupby(group_cols)

        for (sid, serid, condition), group in tqdm(grouped, desc="Processing series groups"):
            row0 = group.iloc[0]
            series_desc = row0["series_description"]
            x_val = row0["x"]
            y_val = row0["y"]
            orig_w = row0["width"]
            orig_h = row0["height"]
            target_instance = row0["instance_number"]

            left, right, top, bottom = self.get_crop_margins(series_desc)
            if (left + right + top + bottom) == 0:
                continue

            out_subfolder = self.get_subfolder_name(condition)
            out_dir = os.path.join(self.output_dir, out_subfolder)
            os.makedirs(out_dir, exist_ok=True)

            dcm_folder = os.path.join(
                self.raw_path, "train_images", str(sid), str(serid))
            dcm_paths = sorted(
                glob(os.path.join(dcm_folder, "*.dcm")),
                key=lambda p: int(os.path.basename(p).replace(".dcm", ""))
            )

            # find target index
            target_index = None
            for i, dp in enumerate(dcm_paths):
                inst_num = int(os.path.basename(dp).replace(".dcm", ""))
                if inst_num == target_instance:
                    target_index = i
                    break
            if target_index is None:
                print(
                    f"[WARN] Missing target slice for study_id={sid}, series_id={serid}")
                continue

            # slices
            if self.preprocessing_mode.lower() == "target_window":
                start_idx = max(target_index - self.slices_before, 0)
                end_idx = min(target_index + self.slices_after,
                              len(dcm_paths)-1)
                selected_paths = dcm_paths[start_idx:end_idx+1]
            else:
                if self.final_depth == 1:
                    selected_paths = [dcm_paths[target_index]]
                else:
                    selected_paths = dcm_paths

            # build volume
            slices = []
            for dp in selected_paths:
                img = self.load_and_normalize_dicom(dp)
                img = self.resize_2d(
                    img, self.cropped_size[0], self.cropped_size[1])
                x_scaled, y_scaled = self.scale_xy(x_val, y_val, orig_w, orig_h,
                                                   self.cropped_size[1], self.cropped_size[0])
                patch = self.crop_around_xy(
                    img, x_scaled, y_scaled, left, right, top, bottom)
                if patch.shape[0] == 0 or patch.shape[1] == 0:
                    continue
                slices.append(patch)

            if len(slices) == 0:
                continue

            # pad
            max_h = max(s.shape[0] for s in slices)
            max_w = max(s.shape[1] for s in slices)
            padded_slices = []
            for s in slices:
                pad_h = max_h - s.shape[0]
                pad_w = max_w - s.shape[1]
                padded = F.pad(s, (0, pad_w, 0, pad_h), "constant", 0.0)
                padded_slices.append(padded)

            vol_3d = torch.stack(padded_slices, dim=0)

            # Depth adjustment
            if self.preprocessing_mode.lower() == "full_series":
                vol_3d = self.pad_or_resize_3d(
                    vol_3d, self.final_depth, self.final_size)
            else:
                # target_window => keep D but fix H,W
                vol_3d = F.interpolate(
                    vol_3d.unsqueeze(0).unsqueeze(0),
                    size=(vol_3d.shape[0],
                          self.final_size[0], self.final_size[1]),
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)

            out_path = os.path.join(out_dir, f"{sid}.pt")
            torch.save(vol_3d, out_path)

        # Step 3: Overwrite merged_train_data.csv with final shape if you want
        # We'll just store the filtered DataFrame now
        df_filtered.to_csv(merged_csv_path, index=False)
        print(
            f"[INFO] Overwrote merged_train_data.csv with final shape (only target disc + no Axial T2): {df_filtered.shape}")
        print(f"[INFO] Tensors saved in subfolders under: {self.output_dir}")


if __name__ == "__main__":
    preprocessor = DataPreprocessor("config.yml")
    preprocessor.process()
