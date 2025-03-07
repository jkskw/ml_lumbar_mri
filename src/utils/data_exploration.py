import matplotlib.pyplot as plt
import numpy as np
import torch

class DatasetExplorer:
    """
    A class that provides dataset information and sample slice visualization
    for both single-disease and multi-disease classification modes.
    """

    def __init__(self,
                 train_ds,
                 val_ds,
                 test_ds,
                 final_depth,
                 final_size,
                 classification_mode="single_binary",
                 disease=None):
        """
        Args:
            train_ds: Training dataset (SingleDiseaseDataset or MultiLabelSpinalDataset).
            val_ds:   Validation dataset.
            test_ds:  Test dataset.
            final_depth (int): Depth dimension used for building the dataset.
            final_size (tuple[int,int]): (height, width).
            classification_mode (str): e.g. "single_multiclass", "single_binary", "multi_multiclass", "multi_binary".
            disease (str, optional): For single-disease classification, name of the disease, e.g. "scs".
        """
        self.train_ds = train_ds
        self.val_ds   = val_ds
        self.test_ds  = test_ds
        self.final_depth = final_depth
        self.final_size  = final_size
        self.classification_mode = classification_mode
        self.disease = disease  # only relevant if single

    def show_dataset_info(self):
        """
        Prints the number of samples in train/val/test sets and basic classification mode info.
        """
        print(f"[INFO] Classification Mode : {self.classification_mode}")
        if self.disease is not None:
            print(f"[INFO] Disease (single)  : {self.disease}")

        print(f"[INFO] Number of samples in the training set   : {len(self.train_ds)}")
        print(f"[INFO] Number of samples in the validation set : {len(self.val_ds)}")
        print(f"[INFO] Number of samples in the test set       : {len(self.test_ds)}")
        print(f"[INFO] final_depth = {self.final_depth}, final_size = {self.final_size}")
        print("=================================================")

    def visualize_sample_slices(self, num_slices=5, sample_idx=0):
        """
        Visualizes a subset of depth slices from a single sample in the training dataset.
        
        Args:
            num_slices (int): Number of slices (depth dimension) to display horizontally.
            sample_idx (int): Index of sample in train_ds to visualize.
        """
        sample_volume, sample_label = self.train_ds[sample_idx]
        # sample_volume shape => [channels, depth, H, W]
        # For single disease => channels=1
        # For multi disease  => channels=3  (scs, lnfn, rnfn)

        print(f"=== Sample Visualization: train_ds[{sample_idx}] ===")
        print(f"Volume shape (channels, D, H, W): {sample_volume.shape}")
        
        # If single binary => float label, single multiclass => int label
        # If multi => shape [3], e.g. [scs_label, lnfn_label, rnfn_label]
        if torch.is_tensor(sample_label):
            if sample_label.ndim == 0:
                # single scalar label
                label_value = sample_label.item()
                print(f"Example label: {label_value}")
            else:
                # e.g. multi-disease => [3]
                label_value = sample_label.cpu().numpy()
                print(f"Example multi-label: {label_value}")
        else:
            print(f"Example label: {sample_label}")

        # Convert to NumPy
        sample_np = sample_volume.numpy()
        depth_dim = sample_np.shape[1]
        slices_to_show = min(num_slices, depth_dim)

        # Plot the slices horizontally
        fig, axes = plt.subplots(1, slices_to_show, figsize=(3*slices_to_show, 3))
        if slices_to_show == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(sample_np[0, i], cmap="gray")  # channel=0 => scs or single disease
            ax.set_title(f"Slice {i}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()
