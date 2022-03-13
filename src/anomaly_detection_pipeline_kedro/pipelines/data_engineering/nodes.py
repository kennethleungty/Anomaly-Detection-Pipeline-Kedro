"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.17.7
"""

from typing import Any, Callable, Dict
import pandas as pd
from datetime import timedelta, datetime as dt


def merge_data(partitioned_input: Dict[str, Callable[[], Any]]) -> pd.DataFrame:
    """Concatenate input partitions into one pandas DataFrame.

    Args:
        partitioned_input: A dictionary with partition ids as keys and load functions as values.

    Returns:
        Pandas DataFrame representing a concatenation of all loaded partitions.
    """
    merged_df = pd.DataFrame()

    for partition_id, partition_load_func in sorted(partitioned_input.items()):
        partition_data = partition_load_func()  # load actual partition data
        merged_df = pd.concat([merged_df, partition_data], ignore_index=True, sort=True) # concat with existing result

    return merged_df


def process_data(merged_df: pd.DataFrame, predictor_cols: list) -> pd.DataFrame:
    """Process the merged dataset

    Args:
        merged_df (pd.DataFrame): Dataframe containing the consolidated credit card transaction data

    Returns:
        pd.DataFrame: Pandas dataframe representing the processed dataset
    """
    # Generate date column
    merged_df['TX_DATETIME'] =  pd.to_datetime(merged_df['TX_DATETIME'], infer_datetime_format=True)
    merged_df['TX_DATE'] = merged_df['TX_DATETIME'].dt.date

    # Only keep columns which are meaningful and predictive (based on domain knowledge)
    processed_df = merged_df[predictor_cols]

    return processed_df


def train_test_split(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Split processed dataset in train and test sets

    Args:
        processed_df (pd.DataFrame): Dataframe containing the processed transaction dataset

    Returns:
        Pandas dataframes of the training data, test data, and test labels (if any)
    """
    # Perform chronological train test split (80:20) i.e. 8 weeks:2 weeks
    processed_df['TX_DATE'] =  pd.to_datetime(processed_df['TX_DATE'], infer_datetime_format=True)
    split_date = processed_df['TX_DATE'].min() + timedelta(days=(8*7)) 
    train_df = processed_df.loc[processed_df['TX_DATE'] <= split_date]
    test_df = processed_df.loc[processed_df['TX_DATE'] > split_date]

    # Drop date column
    train_df.drop(columns=['TX_DATE'], inplace=True)
    test_df.drop(columns=['TX_DATE'], inplace=True)

    # Drop actual label in dataset if any (supposed to be unsupervised training)
    if 'TX_FRAUD' in train_df.columns:
        train_df = train_df.drop(columns=['TX_FRAUD'])
    
    # Store test labels (if any) for subsequent model evaluation
    if 'TX_FRAUD' in test_df.columns:
        test_labels = test_df[['TX_FRAUD']]
        test_df = test_df.drop(columns=['TX_FRAUD'])
    else:
        test_labels = pd.DataFrame() # Empty dataframe if no test labels present

    return train_df, test_df, test_labels