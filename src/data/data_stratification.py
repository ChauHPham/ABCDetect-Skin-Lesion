import pandas as pd
import matplotlib.pyplot as plt

def get_stratified_sampling_df(df, stratify_by, split_ratios=[0.8, 0.1, 0.1]):
    """
    Splits a DataFrame into stratified subsets based on class distribution.

    Parameters:
        df (pd.DataFrame): The input DataFrame to split.
        stratify_by (str): Column name to stratify by.
        split_ratios (tuple): A tuple of n floats such as (train, val, test) or (train, test) that must sum to 1.0.

    Returns:
        tuple: (length based on # of split percentages passed)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`dataframe` must be a pandas DataFrame.")
    
    if not stratify_by:
        raise ValueError("`stratified_column_key` must not be empty.")
    if stratify_by not in df.columns:
        raise KeyError(f"`{stratify_by}` is not a column in the dataframe.")
    
    if sum(split_ratios) != 1.0:
        raise ValueError(f"`split_percentages` must sum to 1.0 (currently sums to {sum(split_ratios)})")

    # Setup
    classes = df[stratify_by].unique()
    splits = [[] for _ in split_ratios]

    for label in classes:
        # Sample randomly in order to shuffle any implicit sorting done by the dataset
        df_subset = df.loc[df[stratify_by] == label].sample(frac=1, random_state=42)

        # Partition the dataframe based on the split ratios
        total = len(df_subset)
        start = 0
        for idx, percentage in enumerate(split_ratios):
            end = total if idx == len(split_ratios) - 1 else start + round(total * percentage)
            splits[idx].append(df_subset.iloc[start:end])
            start = end 
    
    final_splits = []
    for partition in splits:
        final_splits.append(pd.concat(partition).reset_index(drop=True))

    return final_splits

def visualize_dataframe_column_as_histogram(df):
    lesion_histogram = df['dx'].hist(bins=7)
    lesion_histogram.set_title("Lesion Type Histogram")
    lesion_histogram.set_xlabel("Lesion Type")
    lesion_histogram.set_ylabel("Image Count in HAM 10000 database")
    plt.show()