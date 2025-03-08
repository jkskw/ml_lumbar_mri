import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_training_curves(
    train_losses, val_losses,
    train_accuracies=None, val_accuracies=None,
    figsize=(10,5)
):
    """
    Plots training and validation loss, plus optional accuracy curves.
    
    Args:
        train_losses (list[float]): Training loss values over epochs.
        val_losses (list[float]): Validation loss values over epochs.
        train_accuracies (list[float], optional): Training accuracy values. Default None.
        val_accuracies (list[float], optional): Validation accuracy values. Default None.
        figsize (tuple): Figure size, e.g. (10, 5).
    """
    num_epochs = len(train_losses)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # --- Loss subplot ---
    axes[0].plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    axes[0].plot(range(1, num_epochs+1), val_losses,   label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # --- Accuracy subplot ---
    if (train_accuracies is not None) and (val_accuracies is not None):
        axes[1].plot(range(1, num_epochs+1), train_accuracies, label="Train Acc")
        axes[1].plot(range(1, num_epochs+1), val_accuracies,   label="Val Acc")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True)
    else:
        # If accuracy is not provided, we can turn off the second subplot or show an empty plot
        axes[1].text(0.5, 0.5, "No accuracy data", ha='center', va='center', fontsize=12)
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", figsize=(6,6)):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=figsize)
    
    # If no class names provided, create default ones from the shape of cm.
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()