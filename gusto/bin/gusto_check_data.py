'''Script that checks data integrity, and creates the dataseet where applicable.'''

import pickle

import numpy as np
import pandas as pd

from gusto.lib.data import DATA_DIR, TEST_DATA
from gusto.lib.data import TRAINING_DATA, TRAINING_DATA_NOMISSING
from gusto.lib.data import VALIDATION_DATA, VALIDATION_DATA_NOMISSING
from gusto.lib.data import FULL_3M_DATA, FULL_9M_DATA, FULL_48M_DATA


def create_autoencoder_dataset():
    '''Create the dataset for the autoencoder.'''
    source_data = DATA_DIR.joinpath('3-9-48-72months_383CpGs_153indivs_age_related.csv')
    if not source_data.exists():
        raise FileNotFoundError(
            f'{source_data} not found, please move the data to the correct location.'
        )
    if not TRAINING_DATA.exists() or not VALIDATION_DATA.exists() or not TEST_DATA.exists():
        print('Creating the dataset...')
        df = pd.read_csv(source_data)
        create_dataset(df)
    if not TRAINING_DATA_NOMISSING.exists() or not VALIDATION_DATA_NOMISSING.exists():
        print('Creating the dataset with no missing values...')
        df = pd.read_csv(source_data)
        create_nonzero_dataset(df)
    return True


def check_full_dataset():
    '''Check the full dataset.'''
    for dataset in [FULL_3M_DATA, FULL_9M_DATA, FULL_48M_DATA]:
        if not dataset.exists():
            raise FileNotFoundError(
                f'{dataset} not found, please move the data to the correct location.'
            )
    return True


def convert_df_to_dict(df: pd.DataFrame, unique_ids, sorted_probeids=None, sorted_months=None):
    '''Convert the DataFrame to a dictionary.'''
    filtered_df = df[df['ID'].isin(unique_ids)]

    # Get unique ProbeIDs and timepoints
    sorted_probeids = (
        sorted_probeids if sorted_probeids is not None else np.sort(df['ProbeID'].unique())
    )
    sorted_months = sorted_months if sorted_months is not None else np.sort(df['Months'].unique())

    # Create a dictionary to store the data
    data = {}

    # Create a dictionary to map ProbeID and Months to their indices
    probeid_to_idx = {probeid: idx for idx, probeid in enumerate(sorted_probeids)}
    month_to_idx = {month: idx for idx, month in enumerate(sorted_months)}

    # Initialize arrays in the dictionary for each ID
    for id_val in unique_ids:
        data[id_val] = {}
        data[id_val]['data'] = np.zeros(
            (len(sorted_probeids), len(sorted_months)), dtype=np.float32
        )
        data[id_val]['mask'] = np.zeros((len(sorted_probeids), len(sorted_months)), dtype=int)
        data[id_val]['time'] = np.zeros((len(sorted_months), 1), dtype=np.float32)

    # Fill in the array with Output values
    for _, row in filtered_df.iterrows():
        id_val, probe_id, month, inputs, output = (
            row['ID'],
            row['ProbeID'],
            row['Months'],
            row['Input'],
            row['Output'],
        )

        # Get the index for ProbeID and Months
        probe_idx = probeid_to_idx[probe_id]
        month_idx = month_to_idx[month]

        if np.isnan(output):
            continue

        # Assign the output value to the correct position
        data[id_val]['data'][probe_idx, month_idx] = output
        data[id_val]['mask'][probe_idx, month_idx] = 1
        if data[id_val]['time'][month_idx] == 0:
            data[id_val]['time'][month_idx] = inputs

    return {'data': data, 'probeids': sorted_probeids, 'months': sorted_months}


def create_dataset(df: pd.DataFrame):
    '''Create the dataset for the training, validation, and test sets.'''
    # create an rng state
    rng = np.random.default_rng(42)
    # Get the unique IDs within the df
    unique_subj_ids = df['ID'].unique()
    # Shuffle the unique IDs
    rng.shuffle(unique_subj_ids)
    # Split into train (80%), validation (10%), and test (10%) sets
    train_ids = unique_subj_ids[: int(0.8 * len(unique_subj_ids))]
    val_ids = unique_subj_ids[int(0.8 * len(unique_subj_ids)) : int(0.9 * len(unique_subj_ids))]
    test_ids = unique_subj_ids[int(0.9 * len(unique_subj_ids)) :]
    # Create a data structure consisting of a dictionary of ids, ProbeID, and timepoints
    # for each dataset
    train_dict = convert_df_to_dict(df, train_ids)
    val_dict = convert_df_to_dict(df, val_ids, train_dict['probeids'], train_dict['months'])
    test_dict = convert_df_to_dict(df, test_ids, train_dict['probeids'], train_dict['months'])

    # Save the data structure to a pickle file
    with open(TRAINING_DATA, 'wb') as f:
        pickle.dump(train_dict, f)
    with open(VALIDATION_DATA, 'wb') as f:
        pickle.dump(val_dict, f)
    with open(TEST_DATA, 'wb') as f:
        pickle.dump(test_dict, f)


def create_nonzero_dataset(df: pd.DataFrame):
    '''Create the dataset for the training, validation, and test sets with no missing values.'''
    # Step 1: Check if each unique ID has all 4 months
    months = {3, 9, 48, 72}
    valid_ids_months = df.groupby('ID')['Months'].apply(lambda x: set(x) == months)
    # Step 2: Check if each unique ID has all 383 unique ProbeIDs
    unique_probeids = set(df['ProbeID'].unique())
    valid_ids_probeids = df.groupby('ID')['ProbeID'].apply(lambda x: set(x) == unique_probeids)
    # Step 3: Check if each unique ID has non-zero values for Input & Output
    valid_ids_non_zero_input = df.groupby('ID')['Input'].apply(lambda x: (x > 0).all())
    valid_ids_non_zero_output = df.groupby('ID')['Output'].apply(lambda x: (x > 0).all())
    valid_ids = (
        valid_ids_months & valid_ids_probeids & valid_ids_non_zero_input & valid_ids_non_zero_output
    )
    valid_ids.sum()

    # create an rng state
    rng = np.random.default_rng(42)
    # Get the unique IDs within the df
    unique_subj_ids = df['ID'].unique()[valid_ids]

    # Shuffle the unique IDs
    rng.shuffle(unique_subj_ids)
    # Split into train (80%), validation (10%), and test (10%) sets
    train_ids = unique_subj_ids[: int(0.9 * len(unique_subj_ids))]
    val_ids = unique_subj_ids[int(0.9 * len(unique_subj_ids)) :]
    # Create a data structure consisting of a dictionary of ids, ProbeID, and timepoints
    # for each dataset
    train_dict = convert_df_to_dict(df, train_ids)
    val_dict = convert_df_to_dict(df, val_ids, train_dict['probeids'], train_dict['months'])

    # Save the data structure to a pickle file
    with open(TRAINING_DATA_NOMISSING, 'wb') as f:
        pickle.dump(train_dict, f)
    with open(VALIDATION_DATA_NOMISSING, 'wb') as f:
        pickle.dump(val_dict, f)


if __name__ == '__main__':
    if create_autoencoder_dataset():
        print('Small dataset okay.')
    if check_full_dataset():
        print('Large dataset okay.')
