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
    A class to preprocess lumbar MRI DICOM images by loading, normalizing,
    cropping, resizing, and stacking them into 3D volumes (tensors) for model training.

    Supports two preprocessing modes:
      - "absolute": Build tensor from all slices in the series and then force the volume
         to have a specific final_depth via padding/interpolation.
      - "context": Build tensor from a window of slices around the target instance.
         For example, with slices_before=2 and slices_after=2, if target instance is 8,
         the tensor will consist of slices [6, 7, 8, 9, 10].

    The processing parameters are loaded from a YAML configuration file.
    """

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        # Use raw_path from the nested "data" section
        self.raw_path = self.config["data"].get("raw_path", "./data/raw")
        # Use the tensors_path from the "data" section for output volumes
        self.tensors_base_path = self.config["data"].get(
            "tensors_path", "./data/processed/tensors")

        # Preprocessing settings are under the "preprocessing" key
        self.target_disc = self.config["preprocessing"].get(
            "target_disc", "L4/L5")
        self.final_depth = self.config["preprocessing"].get("final_depth", 32)
        self.final_size = tuple(
            self.config["preprocessing"].get("final_size", [96, 96]))
        self.cropped_size = tuple(
            self.config["preprocessing"].get("cropped_size", [512, 512]))
        self.cropping_dict = self.config["preprocessing"].get("cropping", {})

        # Determine which preprocessing mode to use: "absolute" or "context"
        self.preprocessing_mode = self.config.get(
            "preprocessing_mode", "absolute")
        # For context mode, these parameters determine how many slices before and after the target slice to include
        context_cfg = self.config.get("context", {})
        self.slices_before = context_cfg.get("slices_before", 0)
        self.slices_after = context_cfg.get("slices_after", 0)

        # Build the output directory name based on parameters
        self.output_dir = os.path.join(
            self.tensors_base_path,
            f"{self.final_depth}_{self.final_size[0]}x{self.final_size[1]}_{self.preprocessing_mode}_{self.slices_before}s{self.slices_after}"
        )

    def load_config(self, config_path: str) -> dict:
        """
        Load the YAML configuration file.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print("[INFO] Loaded configuration:")
        print(config)
        return config

    def merge_csv_data(self) -> pd.DataFrame:
        """
        Merge the CSV files (train_series_descriptions.csv and train_label_coordinates.csv)
        from the raw data directory and compute the DICOM shape (height, width) for each row.
        The merged CSV is saved to a file named 'merged_train_data.csv' in the raw data directory.
        If the file already exists, it is loaded instead.
        """
        merged_csv_path = os.path.join(self.raw_path, "merged_train_data.csv")
        if os.path.exists(merged_csv_path):
            print(f"[INFO] Merged CSV already exists at {merged_csv_path}.")
            return pd.read_csv(merged_csv_path)

        # Load the individual CSV files
        series_csv = os.path.join(
            self.raw_path, "train_series_descriptions.csv")
        labels_csv = os.path.join(self.raw_path, "train_label_coordinates.csv")
        df_series = pd.read_csv(series_csv)
        df_labels = pd.read_csv(labels_csv)
        # Merge them on study_id and series_id
        merged_df = pd.merge(df_labels, df_series, on=[
                             "study_id", "series_id"], how="inner")
        print(
            f"[INFO] Merged DataFrame shape (before adding DICOM shape): {merged_df.shape}")

        # Compute DICOM shape for each row (using instance_number from labels)
        shapes = []
        for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Computing DICOM shapes"):
            h, w = self.get_dicom_shape(
                row["study_id"], row["series_id"], row["instance_number"])
            shapes.append((h, w))
        merged_df["height"] = [s[0] for s in shapes]
        merged_df["width"] = [s[1] for s in shapes]
        # Drop rows with missing shapes
        merged_df.dropna(subset=["height", "width"], inplace=True)
        merged_df["height"] = merged_df["height"].astype(int)
        merged_df["width"] = merged_df["width"].astype(int)
        # Save the merged CSV
        merged_df.to_csv(merged_csv_path, index=False)
        print(
            f"[INFO] Merged CSV created at {merged_csv_path} with shape: {merged_df.shape}")
        return merged_df

    def get_dicom_shape(self, study_id: int, series_id: int, instance_number: int):
        """
        Reads a DICOM file and returns its (height, width).
        """
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
        """
        Load a DICOM file, clamp pixel intensities between the 1st and 99th percentiles,
        and normalize the image to the [0, 1] range.
        Returns a tensor of shape [H, W].
        """
        ds = dcm.dcmread(dicom_path)
        arr = ds.pixel_array.astype(np.float32)
        t = torch.from_numpy(arr)
        p01 = torch.quantile(t, 0.01)
        p99 = torch.quantile(t, 0.99)
        t = torch.clamp(t, p01, p99)
        t = t - t.min()
        t = t / (t.max() + 1e-6)
        return t

    def resize_2d(self, image: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        """
        Resize a 2D image using bilinear interpolation.
        """
        image = image.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
        resized = F.interpolate(image, size=(
            out_h, out_w), mode='bilinear', align_corners=False)
        return resized.squeeze(0).squeeze(0)

    def crop_around_xy(self, image: torch.Tensor, cx: float, cy: float,
                       left: int, right: int, top: int, bottom: int) -> torch.Tensor:
        """
        Crop the image around the center (cx, cy) using specified margins.
        """
        H, W = image.shape
        row_min = max(int(cy - top), 0)
        row_max = min(int(cy + bottom), H)
        col_min = max(int(cx - left), 0)
        col_max = min(int(cx + right), W)
        if row_min >= row_max or col_min >= col_max:
            return torch.zeros((0, 0), dtype=image.dtype)
        return image[row_min:row_max, col_min:col_max]

    def pad_or_resize_3d(self, volume: torch.Tensor, out_depth: int, final_hw: tuple) -> torch.Tensor:
        """
        Adjust the depth of a 3D volume to out_depth and resize each slice to final_hw.
        """
        D, H, W = volume.shape
        if D < out_depth:
            pad_depth = out_depth - D
            volume = torch.cat([volume, torch.zeros(
                (pad_depth, H, W), dtype=volume.dtype)], dim=0)
        elif D > out_depth:
            volume = F.interpolate(volume.unsqueeze(0).unsqueeze(0),
                                   size=(out_depth, H, W),
                                   mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
        final_h, final_w = final_hw
        volume = F.interpolate(volume.unsqueeze(0).unsqueeze(0),
                               size=(volume.shape[0], final_h, final_w),
                               mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
        return volume

    def scale_xy(self, x: float, y: float, orig_w: int, orig_h: int, new_w: int, new_h: int):
        """
        Scale (x, y) coordinates from original dimensions to new dimensions.
        """
        return x * (new_w / orig_w), y * (new_h / orig_h)

    def get_crop_margins(self, series_desc: str):
        """
        Retrieve cropping margins for the given series description based on configuration.
        """
        if series_desc == "Sagittal T2/STIR":
            margins = self.cropping_dict.get("scs", {}).get("sagt2", {})
        elif series_desc == "Sagittal T1":
            margins = self.cropping_dict.get("nfn", {}).get("sagt1", {})
        else:
            margins = {}
        return margins.get("left", 0), margins.get("right", 0), margins.get("upper", 0), margins.get("lower", 0)

    def get_subfolder_name(self, condition: str) -> str:
        """
        Returns a subfolder name based on the diagnosis condition.
        """
        if condition == "Spinal Canal Stenosis":
            return "scs"
        elif condition in ["Right Neural Foraminal Narrowing", "Left Neural Foraminal Narrowing"]:
            return "nfn"
        return "other"

    def process(self):
        """
        Process raw DICOM data:
          - Merge CSV files (if not already merged) and compute DICOM shapes.
          - Filter data based on target disc and series description.
          - For each (study_id, series_id) group, process DICOM slices:
              - Identify the target slice based on instance_number.
              - Depending on the preprocessing mode:
                  * "absolute": use all slices in the series and later force the volume to have a final_depth.
                  * "context": select slices using slices_before and slices_after relative to the target slice.
              - For each selected slice, load, normalize, resize, crop, and stack.
          - Save the final 3D volume (tensor) to the output directory.
        """
        # Merge CSV data if needed (this creates/loads merged_train_data.csv in raw_path)
        merged_df = self.merge_csv_data()

        # Filter target disc and remove unwanted series descriptions
        df_target = merged_df[merged_df["level"] == self.target_disc].copy()
        df_target = df_target[df_target["series_description"] != "Axial T2"]
        print(f"[INFO] After filtering: {df_target.shape}")

        # Process each series group
        group_cols = ["study_id", "series_id"]
        grouped = df_target.groupby(group_cols)
        for (sid, serid), group in tqdm(grouped, desc="Processing series groups"):
            row0 = group.iloc[0]
            condition = row0["condition"]
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

            # List and sort DICOM file paths for this series
            dcm_folder = os.path.join(
                self.raw_path, "train_images", str(sid), str(serid))
            dcm_paths = sorted(glob(os.path.join(dcm_folder, "*.dcm")),
                               key=lambda p: int(os.path.basename(p).replace(".dcm", "")))

            # Identify the target slice index based on instance_number
            target_index = None
            for idx, dp in enumerate(dcm_paths):
                inst_num = int(os.path.basename(dp).replace(".dcm", ""))
                if inst_num == target_instance:
                    target_index = idx
                    break

            if target_index is None:
                print(
                    f"Warning: Missing target slice for study_id {sid} (instance {target_instance})")
                continue

            # Determine which slices to use based on preprocessing_mode
            if self.preprocessing_mode.lower() == "context":
                # Use a window of slices around the target slice
                start_index = max(target_index - self.slices_before, 0)
                end_index = min(
                    target_index + self.slices_after, len(dcm_paths) - 1)
                selected_paths = dcm_paths[start_index:end_index + 1]
            else:
                # "absolute" mode: use all slices from the series
                selected_paths = dcm_paths

            # Process each selected slice
            slices = []
            for dp in selected_paths:
                img = self.load_and_normalize_dicom(dp)
                img = self.resize_2d(
                    img, self.cropped_size[0], self.cropped_size[1])
                # Scale the (x, y) coordinates to the resized dimensions
                x_scaled, y_scaled = self.scale_xy(x_val, y_val, orig_w, orig_h,
                                                   self.cropped_size[1], self.cropped_size[0])
                patch = self.crop_around_xy(
                    img, x_scaled, y_scaled, left, right, top, bottom)
                if patch.shape[0] == 0 or patch.shape[1] == 0:
                    continue
                slices.append(patch)

            if len(slices) == 0:
                continue

            # Pad slices to ensure uniform dimensions
            max_h = max(s.shape[0] for s in slices)
            max_w = max(s.shape[1] for s in slices)
            padded_slices = []
            for s in slices:
                pad_h = max_h - s.shape[0]
                pad_w = max_w - s.shape[1]
                padded = F.pad(s, (0, pad_w, 0, pad_h), "constant", 0.0)
                padded_slices.append(padded)

            vol_3d = torch.stack(padded_slices, dim=0)  # Shape: [D, H, W]

            if self.preprocessing_mode.lower() == "absolute":
                # In absolute mode, force the volume to have exactly final_depth slices.
                vol_3d = self.pad_or_resize_3d(
                    vol_3d, self.final_depth, self.final_size)
            else:
                # In context mode, the depth is determined by slices_before + slices_after + 1.
                # Optionally, apply resizing if needed.
                pass

            out_path = os.path.join(out_dir, f"{sid}.pt")
            torch.save(vol_3d, out_path)

        print(
            "[INFO] Data processing completed. Check output directory:", self.output_dir)


if __name__ == "__main__":
    # Example usage: ensure that 'config.yml' is at the root or specify the correct path.
    preprocessor = DataPreprocessor("config.yml")
    preprocessor.process()
