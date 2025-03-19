import pandas as pd
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

def apply_imbalance_strategy(
    df: pd.DataFrame,
    label_col: str,
    method: str = "none",
    undersampling_ratio: float = 0.5,
    oversampling_ratio: float = 1.0,
    smote: bool = False,
    is_multilabel: bool = False
) -> pd.DataFrame:
    """
    Applies undersampling, oversampling, or a hybrid approach to the input DataFrame.
    Returns a new DataFrame with balanced classes.
    """
    if method == "none":
        return df

    # Reset the index to use it as a proxy feature for sampling.
    df = df.reset_index(drop=True)
    X = df.index.values.reshape(-1, 1)  # using indices as features
    y = df[label_col].values

    # For SMOTE, warn and switch to RandomOverSampler since synthetic samples cannot be mapped back.
    if smote and not is_multilabel:
        print("[WARNING] SMOTE is not supported when using DataFrame indices. Switching to RandomOverSampler.")
        smote = False

    sampler = None
    if method == "undersampling":
        sampler = RandomUnderSampler(
            sampling_strategy=undersampling_strategy(undersampling_ratio, y, is_multilabel),
            random_state=42
        )
    elif method == "oversampling":
        sampler = RandomOverSampler(
            sampling_strategy=oversampling_strategy(oversampling_ratio, y),
            random_state=42
        )
    elif method == "hybrid":
        # First oversample, then undersample.
        over = RandomOverSampler(
            sampling_strategy=oversampling_strategy(oversampling_ratio, y),
            random_state=42
        )
        under = RandomUnderSampler(
            sampling_strategy=undersampling_strategy(undersampling_ratio, y, is_multilabel),
            random_state=42
        )
        sampler = Pipeline([('over', over), ('under', under)])
    else:
        return df

    X_res, y_res = sampler.fit_resample(X, y)
    # Use the resulting indices (possibly repeated) to create the balanced DataFrame.
    df_balanced = df.iloc[X_res.flatten()].copy()
    df_balanced.reset_index(drop=True, inplace=True)

    print(f"[IMBALANCE] Resampling method={method}: final distribution => {Counter(y_res)}")
    return df_balanced

def undersampling_strategy(undersampling_ratio, y, is_multilabel=False):
    """
    Reduce the count of majority classes by undersampling_ratio while leaving the minority class untouched.
    For a class with original count 'count' (if it is not the minimum), target = int(count * undersampling_ratio),
    but not below the minimum count.
    """
    c = Counter(y)
    min_count = min(c.values())
    sampling_strategy = {}
    for cl, count in c.items():
        if count == min_count:
            sampling_strategy[cl] = count
        else:
            target = int(count * undersampling_ratio)
            if target < min_count:
                target = min_count
            sampling_strategy[cl] = target
    return sampling_strategy

def oversampling_strategy(oversampling_ratio, y):
    """
    Increase the count of minority classes up to a target defined by oversampling_ratio.
    For each class, target = int(max_count * oversampling_ratio) if that is greater than its current count;
    otherwise, leave it unchanged. The majority class remains at its original count.
    """
    c = Counter(y)
    max_count = max(c.values())
    sampling_strategy = {}
    for cl, count in c.items():
        # Compute the desired target based on the ratio.
        desired_target = int(max_count * oversampling_ratio)
        # For minority classes, raise count to the desired target if it's higher than the current count.
        sampling_strategy[cl] = desired_target if desired_target > count else count
    return sampling_strategy
