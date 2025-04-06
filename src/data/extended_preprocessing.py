import os
import yaml
import torch
import numpy as np
import cv2
from tqdm import tqdm

def proposed_mean_median_3d(volume_np: np.ndarray, ksize=3) -> np.ndarray:
    """
    Simple demonstration of combining median + mean in a local window:
    1) Compute median M of the window.
    2) For each pixel p in the window, averageVal(p) = (p + M)/2.
    3) Final center pixel = mean(averageVal(p) for all p in window.
    """
    if volume_np.ndim != 3:
        raise ValueError("Expected 3D array.")
    out_slices = []
    pad = ksize // 2

    D, H, W = volume_np.shape
    for d in range(D):
        slice_2d = volume_np[d]
        norm_8u = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        slice_out = np.zeros_like(slice_2d, dtype=np.float32)

        for r in range(H):
            for c in range(W):
                rmin = max(0, r - pad)
                rmax = min(H, r + pad + 1)
                cmin = max(0, c - pad)
                cmax = min(W, c + pad + 1)

                patch = norm_8u[rmin:rmax, cmin:cmax].flatten()
                median_val = np.median(patch)
                patch_f = patch.astype(np.float32)
                avg_vals = (patch_f + median_val) / 2.0
                center_pix = avg_vals.mean()
                slice_out[r, c] = center_pix / 255.0

        out_slices.append(slice_out)

    return np.stack(out_slices, axis=0)


def load_config(config_path="config.yml"):
    """
    Load the YAML configuration file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_directory(input_dir, processed_path, technique, ksize):
    """
    Process all .pt files under a given input directory and save the processed
    tensors under processed_path preserving the folder structure.
    """
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing files in {root}", leave=False):
            if file.endswith(".pt"):
                input_file_path = os.path.join(root, file)
                try:
                    volume = torch.load(input_file_path)
                except Exception as e:
                    print(f"Failed to load {input_file_path}: {e}")
                    continue

                # Ensure the loaded object is a 3D tensor
                if not isinstance(volume, torch.Tensor):
                    print(f"File {input_file_path} is not a torch.Tensor. Skipping.")
                    continue
                if volume.ndim != 3:
                    print(f"Tensor in {input_file_path} does not have 3 dimensions. Skipping.")
                    continue

                # Convert to NumPy array for processing
                volume_np = volume.numpy()

                # Process using the selected technique
                if technique == "proposed_mean_median_3d":
                    processed_np = proposed_mean_median_3d(volume_np, ksize=ksize)
                else:
                    print(f"Technique {technique} is not implemented. Skipping file {input_file_path}.")
                    continue

                # Convert back to tensor
                processed_tensor = torch.from_numpy(processed_np)

                # Preserve folder structure: get the relative path from the input_dir
                rel_path = os.path.relpath(root, input_dir)
                output_dir = os.path.join(processed_path, rel_path)
                os.makedirs(output_dir, exist_ok=True)

                output_file_path = os.path.join(output_dir, file)
                torch.save(processed_tensor, output_file_path)
                # print(f"Processed and saved: {output_file_path}")


def main():
    # Load configuration
    config = load_config("config.yml")
    
    # Read processing parameters from the configuration
    # Accept either a single string or a list of directories for input_tensor_dir.
    technique = config.get("preprocessing", {}).get("technique", "proposed_mean_median_3d")
    ksize = config.get("preprocessing", {}).get("ksize", 3)
    input_tensor_dir = config.get("preprocessing", {}).get("input_tensor_dir", "./data/interim")
    
    # processed_path is defined under the data section in config.yml.
    processed_path = config.get("data", {}).get("processed_path", "./data/processed")
    os.makedirs(processed_path, exist_ok=True)
    
    # Allow input_tensor_dir to be a single string or a list of strings.
    if isinstance(input_tensor_dir, str):
        input_dirs = [input_tensor_dir]
    elif isinstance(input_tensor_dir, list):
        input_dirs = input_tensor_dir
    else:
        raise ValueError("input_tensor_dir must be either a string or a list of strings.")
    
    # Process each input directory
    for in_dir in input_dirs:
        print(f"Processing directory: {in_dir}")
        process_directory(in_dir, processed_path, technique, ksize)


if __name__ == "__main__":
    main()
