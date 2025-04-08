import matplotlib.pyplot as plt
import pandas as pd


def stratified_sampling(df: pd.DataFrame, stratify_by: str, split_ratios: list[float]) -> list[pd.DataFrame]:
    """Splits a DataFrame into stratified subsets based on the specified column.

    Args:
        df: The input DataFrame to split.
        stratify_by: Column name to stratify by.
        split_ratios: A list of floats representing the split ratios (e.g., [0.8, 0.1, 0.1]), must sum to 1.0.

    Returns:
        A list of DataFrames corresponding to the specified split ratios.

    Raises:
        TypeError: If the input is not a DataFrame.
        ValueError: If `stratify_by` is empty or if `split_ratios` does not sum to 1.0.
        KeyError: If `stratify_by` is not a column in the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    if not stratify_by:
        raise ValueError("`stratify_by` must not be empty.")
    if stratify_by not in df.columns:
        raise KeyError(f"`{stratify_by}` is not a column in the DataFrame.")

    if abs(sum(split_ratios) - 1.0) > 1e-5:
        raise ValueError(f"`split_ratios` must sum to 1.0 (currently sums to {round(sum(split_ratios), 2)}).")

    # Setup
    categories = df[stratify_by].unique()
    splits = [[] for _ in split_ratios]

    for category in categories:
        # Get a subset of the DataFrame for the current category, then shuffling it
        df_subset = df.loc[df[stratify_by] == category].sample(frac=1)

        # Partition the dataframe based on the split ratios
        total_rows = len(df_subset)
        start = 0
        for idx, ratio in enumerate(split_ratios):
            end = total_rows if idx == len(split_ratios) - 1 else start + round(total_rows * ratio)
            splits[idx].append(df_subset.iloc[start:end])
            start = end

    # Concatenate the partitions into final DataFrames
    final_splits = []
    for partition in splits:
        final_splits.append(pd.concat(partition).reset_index(drop=True))

    return final_splits


def visualize_dx_column_as_histogram(df: pd.DataFrame) -> None:
    """Visualizes the dx column as a histogram.

    Args:
        df: The input DataFrame to visualize.
    """
    lesion_histogram = df["dx"].hist(bins=df["dx"].nunique())
    lesion_histogram.set_title("Lesion Type Histogram")
    lesion_histogram.set_xlabel("Lesion Type")
    lesion_histogram.set_ylabel("Image Count in HAM 10000 database")
    plt.show()
